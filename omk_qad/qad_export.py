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
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .fakequant import FLOAT_TYPES, Codec
from .gguf_export import _import_llama_cpp

log = logging.getLogger("omk_qad.qad_export")


def export_gguf(model: torch.nn.Module, layout: dict, out: Path, llama_cpp: Path | None = None,
                device: str | torch.device = "cpu") -> dict:
    """Write ``model``'s current weights as a GGUF at ``layout``'s plan; returns the export report."""
    t0 = time.time()
    llama_cpp = Path(llama_cpp or layout["llama_cpp"])
    gguf, get_model_architecture, get_model_class, text_type = _import_llama_cpp(llama_cpp)
    from gguf.lazy import LazyBase

    codec = Codec(llama_cpp)
    model_dir = Path(layout["model_dir"])
    hparams = json.loads((model_dir / "config.json").read_text())
    base_cls = get_model_class(get_model_architecture(hparams, text_type))
    named = dict(model.named_parameters())
    if {n: list(p.shape) for n, p in named.items()} != layout["params"]:
        raise SystemExit("model parameters do not match the layout's (names or shapes)")
    names = list(named)

    def tensor_of(name):
        return named[name].detach().to("cpu", torch.float32)

    class LiveModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch
        no_mtp = True  # the text causal LM has no MTP head (gap: graft it from the base)

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
        payload = codec.encode(t["type"], x.to(device))
        decoded = codec.decode(t["type"], payload, rows_n, cols, "cpu").numpy()
        raw = codec.to_bytes(t["type"], payload, rows_n).reshape(*data.shape[:-1], -1)
        real_add(name, raw, raw_dtype=codec.qtype(t["type"]))
        expected[name] = hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
        report["tensors"][name] = {"type": t["type"], "how": how}

    writer.add_tensor = add_tensor
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
    out.with_suffix(".export.json").write_text(json.dumps(report, indent=1))
    if bad or wrong_type:
        raise SystemExit(f"EXPORT VERIFY FAILED: {len(bad)} decode mismatches, {len(wrong_type)} type mismatches")
    return report
