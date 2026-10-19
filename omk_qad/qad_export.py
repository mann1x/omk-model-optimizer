# SPDX-License-Identifier: Apache-2.0
"""Trained latents -> GGUF at the layout's plan, encoded with the encoders training served.

llama.cpp's converter runs in-process on the model's CURRENT tensors (no checkpoint is written in between),
so its renames, splits and head permutations stay its own. Each GGUF tensor it emits is then:

  * float in the plan  -> written as the converter produced it (norms, router, conv, ...);
  * quantized, layout "direct" -> first PROVEN to be the latent rows/columns the trainer encoded
    (bitwise), then encoded with the same codec (ModelOpt for IQ1/IQ2/Q8_0, libggml otherwise);
  * quantized, layout unsupported -> encoded from the converter's values (export-only) and reported.

Finally the GGUF is re-read and every quantized tensor is decoded with gguf-py's reference decoders and
compared with the decode computed here, and every type is checked against the plan.
"""

import hashlib
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .fakequant import FLOAT_TYPES, TERNARY_TYPES, Codec
from .gguf_export import _import_llama_cpp
from .imatrix import mean_sq

log = logging.getLogger("omk_qad.qad_export")


def _encode_rows(codec, qtype: str, x: torch.Tensor, eids: torch.Tensor, msq: torch.Tensor | None,
                 colperm: torch.Tensor | None, device):
    """Encode ``x`` [rows, cols] per run of equal expert id with that expert's importance (as training did)."""
    from .imatrix import importance_for

    cut = [0, *(torch.nonzero(eids[1:] != eids[:-1]).flatten() + 1).tolist(), len(eids)]
    raws, decs = [], []
    for a, b in itertools.pairwise(cut):
        imp = None
        if msq is not None:
            m = msq[int(eids[a]) if msq.shape[0] > 1 else 0]
            imp = importance_for(qtype, m[colperm] if colperm is not None else m)
        payload = codec.encode(qtype, x[a:b].to(device), imp)
        raws.append(codec.to_bytes(qtype, payload, b - a))
        decs.append(codec.decode(qtype, payload, b - a, x.shape[1], "cpu").numpy())
    return np.concatenate(raws), np.concatenate(decs)


def _ternary_tensor(name, t, x, how, folded, codec, layout, imatrix_folded):
    """GGUF bytes and folded-basis values of one ternary target.

    Trained (direct mapping onto a folded latent): the latent's own payloads, i.e. exactly what training served
    (re-folding the converter's values would round in fp32 and could flip a trit at the top-K threshold).
    Export-only: fold the converter's values with the model's basis and ternarize them here.
    """
    from .imatrix import importance_for, mean_sq
    from .ternary import HadamardBasis

    if t["mapping"] == "direct" and t["src"] in folded:
        lat = folded[t["src"]]
        chunks = lat.payloads()[name]
        sizes = [rows.stop - rows.start if isinstance(rows, slice) else len(rows)
                 for seg in lat.segments if seg.gguf == name for rows, _, _ in seg.chunks]
        raw = np.concatenate([codec.to_bytes(t["type"], c, n) for c, n in zip(chunks, sizes, strict=True)])
        dec = np.concatenate([codec.decode(t["type"], c, n, lat.cols, "cpu").numpy()
                              for c, n in zip(chunks, sizes, strict=True)])
        return raw, dec
    cols = x.shape[-1]
    y = HadamardBasis(cols, block=layout["hadamard_block"]).fold(x)
    entries = (imatrix_folded or {}).get("entries", {})
    m = mean_sq(entries[t["src"]])[0] if t["src"] in entries and entries[t["src"]]["sums"].shape[0] == 1 else None
    payload = codec.encode(t["type"], y, importance_for(t["type"], m))
    return codec.to_bytes(t["type"], payload, y.shape[0]), codec.decode(t["type"], payload, y.shape[0], cols,
                                                                          "cpu").numpy()


def _prism_kv(writer, targets: dict, block: int) -> dict:
    """``prism.hadamard.*`` metadata of the ternary targets, exactly as the runtime loader reads it (0588)."""
    from .ternary import seeded_signs

    tern = sorted(g for g, t in targets.items() if t["type"] in TERNARY_TYPES)
    tied = "token_embd.weight" in tern and "output.weight" not in targets
    weights = [g for g in tern if g != "token_embd.weight"]
    inverse = ["token_embd.weight"] if "token_embd.weight" in tern else []
    if not weights:  # the loader refuses an empty list, and token_embd is never a weight_names entry
        raise SystemExit("prism.hadamard needs at least one folded matmul weight besides token_embd")
    widths = sorted({targets[g]["shape"][-1] for g in tern})
    signs = [int(v) for w in widths for v in seeded_signs(w).tolist()]
    kv = {"version": 2 if tied else 1, "tied_output": tied, "block_size": block, "weight_names": weights,
          "inverse_weight_names": inverse, "sign_widths": widths}
    writer.add_uint32("prism.hadamard.version", kv["version"])
    writer.add_bool("prism.hadamard.tied_output", tied)
    writer.add_uint32("prism.hadamard.block_size", block)
    writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
    writer.add_string("prism.hadamard.axis", "input-last-dimension")
    writer.add_string("prism.hadamard.sign_mode", "explicit")
    writer.add_array("prism.hadamard.weight_names", weights)
    if inverse:
        writer.add_array("prism.hadamard.inverse_weight_names", inverse)
    writer.add_array("prism.hadamard.sign_widths", widths)
    writer.add_array("prism.hadamard.sign_values", signs)
    return kv


def export_gguf(model: torch.nn.Module, layout: dict, out: Path, llama_cpp: Path | None = None,
                device: str | torch.device = "cpu", imatrix: dict | None = None, latents: dict | None = None,
                imatrix_folded: dict | None = None) -> dict:
    """Write ``model``'s current weights as a GGUF at ``layout``'s plan; returns the export report.

    ``imatrix`` (omk_qad.imatrix.load record: entries by HF name + metadata) weights the encoders exactly as in
    training, and is saved next to the GGUF in llama.cpp's format (``<stem>.imatrix.gguf``).
    """
    t0 = time.time()
    llama_cpp = Path(llama_cpp or layout["llama_cpp"])
    gguf, get_model_architecture, get_model_class, text_type = _import_llama_cpp(llama_cpp)
    from gguf.lazy import LazyBase

    codec = Codec(llama_cpp)
    ternary = any(t["type"] in TERNARY_TYPES for t in layout["targets"].values())
    if ternary:
        from .ternary import register_gguf_types

        register_gguf_types(gguf)
    folded = {n: lat for n, lat in (latents or {}).items() if lat.basis is not None}
    model_dir = Path(layout["model_dir"])
    hparams = json.loads((model_dir / "config.json").read_text())
    base_cls = get_model_class(get_model_architecture(hparams, text_type))
    named = dict(model.named_parameters())
    if {n: list(p.shape) for n, p in named.items()} != layout["params"]:
        raise SystemExit("model parameters do not match the layout's (names or shapes)")
    names = list(named)

    def tensor_of(name):
        if name in folded:  # a ternary latent is stored folded: the converter must see the model's basis
            return folded[name].model_weight().to("cpu", torch.float32)
        return named[name].detach().to("cpu", torch.float32)

    class LiveModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch
        no_mtp = not any(n.startswith("mtp.") for n in names)  # model.mtp (omk_qad.mtp) -> blk.<n>.nextn.*

        def index_tensors(self, remote_hf_model_id=None):
            from conversion.base import ModelBase

            def live(model_self, remote_hf_model_id=None):
                del remote_hf_model_id
                items = (model_self.filter_tensors((n, lambda n=n: tensor_of(n))) for n in names)
                return dict(item for item in items if item)

            real = ModelBase.index_tensors
            ModelBase.index_tensors = live
            try:
                return super().index_tensors(remote_hf_model_id=remote_hf_model_id)
            finally:
                ModelBase.index_tensors = real

    conv = LiveModel(model_dir, gguf.LlamaFileType.ALL_F32, out, eager=True)
    writer = conv.gguf_writer
    real_add = writer.add_tensor
    targets = layout["targets"]
    report: dict = {"out": str(out), "ftype": layout["ftype"], "tensors": {}}
    expected: dict[str, str] = {}

    def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        data = LazyBase.to_eager(tensor) if isinstance(tensor, LazyBase) else np.asarray(tensor)
        t = targets.get(name)
        if t is None:
            raise SystemExit(f"{name}: not in the layout (layout built from a different model/converter?)")
        if list(data.shape) != t["shape"]:
            raise SystemExit(f"{name}: converter shape {list(data.shape)} != layout {t['shape']}")
        if t["type"] in FLOAT_TYPES:
            real_add(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype, tensor_endianess=tensor_endianess)
            report["tensors"][name] = {"type": t["type"], "how": "float"}
            return
        rows_n, cols = int(np.prod(data.shape[:-1])), int(data.shape[-1])
        x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32).reshape(rows_n, cols))
        how = "export-only"
        if t["mapping"] == "direct":
            src = tensor_of(t["src"]).reshape(-1, cols)[t["rows"].long()]
            if t["colperm"] is not None:
                src = src[:, t["colperm"].long()]
            if not torch.equal(src, x):
                raise SystemExit(f"{name}: converter values differ from the layout's latent rows/columns")
            how = "trained-blocks"
        if t["type"] in TERNARY_TYPES:
            raw, decoded = _ternary_tensor(name, t, x, how, folded, codec, layout, imatrix_folded)
            real_add(name, raw.reshape(*data.shape[:-1], -1), raw_dtype=codec.qtype(t["type"]))
            expected[name] = hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
            report["tensors"][name] = {"type": t["type"], "how": how, "imatrix": "folded" if
                                       (t["src"] in folded and folded[t["src"]].msq is not None) else False}
            return
        src_shape = layout["params"].get(t["src"]) if t["src"] else None
        if t["mapping"] == "direct" and len(src_shape) == 3:
            eids = t["rows"].long() // src_shape[1]
        else:
            eids = torch.zeros(rows_n, dtype=torch.long)
        entries = imatrix["entries"] if imatrix else {}
        msq = mean_sq(entries[t["src"]]) if t["src"] in entries else None
        perm = t["colperm"].long() if t["colperm"] is not None else None
        raw, decoded = _encode_rows(codec, t["type"], x, eids, msq, perm, device)
        raw = raw.reshape(*data.shape[:-1], -1)
        report["tensors"].setdefault(name, {})["imatrix"] = msq is not None
        real_add(name, raw, raw_dtype=codec.qtype(t["type"]))
        expected[name] = hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
        report["tensors"][name] = {"type": t["type"], "how": how, "imatrix": msq is not None}

    writer.add_tensor = add_tensor
    if ternary:
        report["prism_hadamard"] = _prism_kv(writer, targets, layout["hadamard_block"])
    conv.write()
    log.info("written %s in %.1fs", out, time.time() - t0)

    reader = gguf.GGUFReader(out)
    bad, wrong_type, checked = [], [], 0
    for t in reader.tensors:
        planned = targets[t.name]["type"]
        if t.tensor_type.name != planned:
            wrong_type.append(f"{t.name}: {t.tensor_type.name} != plan {planned}")
        if t.name not in expected:
            continue
        if t.tensor_type.name in TERNARY_TYPES:  # gguf-py has no decoder for these: ours (bit-exact, T1)
            from .ternary import decode, unpack

            raw = np.asarray(t.data).reshape(-1, t.data.shape[-1])
            values = decode(*unpack(t.tensor_type.name, raw, int(t.shape[0]))).numpy()
        else:
            values = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        checked += 1
        if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() != expected[t.name]:
            bad.append(t.name)
    counts: dict[str, int] = {}
    for v in report["tensors"].values():
        counts[v["type"]] = counts.get(v["type"], 0) + 1
    report.update(
        verified=checked,
        verify_failures=bad,
        type_mismatches=wrong_type,
        type_counts=counts,
        export_only=[n for n, v in report["tensors"].items() if v["how"] == "export-only"],
        size_bytes=out.stat().st_size,
        seconds=round(time.time() - t0, 1),
    )
    if imatrix:
        from .imatrix import save as save_imatrix

        side = out.with_suffix("")
        side.mkdir(exist_ok=True)
        save_imatrix(imatrix["entries"], layout, side, imatrix["datasets"], imatrix["chunk_size"],
                     imatrix["chunk_count"], llama_cpp)
        (side / "imatrix.gguf").replace(out.with_suffix(".imatrix.gguf"))
        (side / "imatrix.pt").replace(out.with_suffix(".imatrix.pt"))
        side.rmdir()
        report["imatrix"] = str(out.with_suffix(".imatrix.gguf"))
    report["with_imatrix"] = sum(1 for v in report["tensors"].values() if v.get("imatrix"))
    out.with_suffix(".export.json").write_text(json.dumps(report, indent=1))
    if bad or wrong_type:
        raise SystemExit(f"EXPORT VERIFY FAILED: {len(bad)} decode mismatches, {len(wrong_type)} type mismatches")
    return report
