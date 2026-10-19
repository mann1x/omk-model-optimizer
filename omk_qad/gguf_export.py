# SPDX-License-Identifier: Apache-2.0
"""ModelOpt GGML-packed HF checkpoint -> GGUF, carrying the trained packed bytes verbatim.

Upstream convert_hf_to_gguf has no import for ModelOpt's ``packing: ggml`` tensors (U8, shape
[rows, blocks, block_bytes]). This drives llama.cpp's own converter in-process so every architecture rule
(renames, expert stacking, head reorders) stays the converter's, and changes only the tensor payloads:

  1. each packed tensor is handed to the converter as its exact float32 decode (ModelOpt decoder);
  2. at ``add_tensor`` time every row of every 2-D+ output tensor is looked up by hash in the decoded
     source rows; a tensor whose rows all match is written as the SOURCE's packed bytes (raw_dtype = its
     GGML type), so the GGUF holds exactly the blocks QAD trained;
  3. a tensor that does not row-match (the converter permuted its columns) is re-encoded with gguf-py and
     kept only if the re-encode decodes EXACTLY to the converter's values (true for Q8_0 under a
     head-granular column permutation); otherwise it stays float and is reported;
  4. the token embedding (kept BF16 by ModelOpt) is quantized per --embed-type, as the census policy says;
  5. the written GGUF is re-read and every quantized tensor is decoded with gguf-py's reference decoders
     and compared with what the converter produced. Any difference fails the export.

  python -m omk_qad.gguf_export --src PACKED_DIR --out FILE.gguf [--llama-cpp /opt/llama.cpp] [--embed-type q8_0]
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("omk_qad.gguf_export")

GGML_ALGO = {
    "IQ1_S": "iq1_s",
    "IQ1_M": "iq1_m",
    "IQ2_XXS": "iq2_xxs",
    "IQ2_XS": "iq2_xs",
    "IQ2_S": "iq2_s",
    "Q8_0": "q8_0",
}


def _import_llama_cpp(llama_cpp: Path):
    """Import the converter and gguf-py from the llama.cpp tree (never a pip gguf: they must match)."""
    for p in (str(llama_cpp / "gguf-py"), str(llama_cpp)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "gguf" in sys.modules and not str(sys.modules["gguf"].__file__).startswith(str(llama_cpp)):
        raise RuntimeError(f"a different gguf is already imported: {sys.modules['gguf'].__file__}")
    import gguf
    from conversion import get_model_architecture, get_model_class
    from conversion.base import ModelType

    return gguf, get_model_architecture, get_model_class, ModelType.TEXT


class RowIndex:
    """Hash index of the decoded rows of every packed source tensor.

    Two independent 64-bit multiplicative hashes of the float32 row bits; a lookup must agree on both,
    and every match is later proven by exact decode equality, so a collision cannot pass silently.
    """

    def __init__(self, seed: int = 0x5EED):
        """Empty index; hash weights are drawn per row width on first use."""
        self._rng = np.random.default_rng(seed)
        self._weights: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._k1: list[np.ndarray] = []
        self._k2: list[np.ndarray] = []
        self._src: list[np.ndarray] = []
        self._row: list[np.ndarray] = []
        self.sources: list[tuple[str, str]] = []  # (tensor name, ggml format) per source id
        self._sorted = False

    def _hash(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = rows.shape[1]
        if width not in self._weights:
            hi = np.iinfo(np.uint64).max
            w1 = self._rng.integers(1, hi, size=width, dtype=np.uint64) | np.uint64(1)
            w2 = self._rng.integers(1, hi, size=width, dtype=np.uint64) | np.uint64(1)
            self._weights[width] = (w1, w2)
        w1, w2 = self._weights[width]
        bits = np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32).astype(np.uint64)
        width_tag = np.uint64((width * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        with np.errstate(over="ignore"):
            return (bits * w1).sum(axis=1, dtype=np.uint64) ^ width_tag, (bits * w2).sum(axis=1, dtype=np.uint64)

    def add(self, name: str, fmt: str, decoded: np.ndarray, chunk: int = 65536) -> None:
        """Index every row of ``decoded`` (2-D float32) as rows of source ``name``."""
        sid = len(self.sources)
        self.sources.append((name, fmt))
        for start in range(0, decoded.shape[0], chunk):
            part = decoded[start : start + chunk]
            k1, k2 = self._hash(part)
            self._k1.append(k1)
            self._k2.append(k2)
            self._src.append(np.full(part.shape[0], sid, dtype=np.int32))
            self._row.append(np.arange(start, start + part.shape[0], dtype=np.int64))
        self._sorted = False

    def _finalize(self) -> None:
        if self._sorted:
            return
        k1, k2 = np.concatenate(self._k1), np.concatenate(self._k2)
        src, row = np.concatenate(self._src), np.concatenate(self._row)
        order = np.argsort(k1, kind="stable")
        self._k1, self._k2, self._src, self._row = [k1[order]], [k2[order]], [src[order]], [row[order]]
        self._sorted = True

    def lookup(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(source id, source row) per row; source id -1 where the row is not indexed."""
        self._finalize()
        k1s, k2s, srcs, rows_idx = self._k1[0], self._k2[0], self._src[0], self._row[0]
        q1, q2 = self._hash(rows)
        pos = np.searchsorted(k1s, q1)
        pos_c = np.minimum(pos, len(k1s) - 1)
        ok = (k1s[pos_c] == q1) & (k2s[pos_c] == q2)
        return np.where(ok, srcs[pos_c], -1), np.where(ok, rows_idx[pos_c], -1)


def _packed_groups(src: Path) -> dict[str, str]:
    """Map module path -> ggml format, from config.json quantization_config.config_groups."""
    cfg = json.loads((src / "config.json").read_text())
    groups = (cfg.get("quantization_config") or {}).get("config_groups") or {}
    out = {}
    for g in groups.values():
        if g.get("packing") != "ggml":
            continue
        fmt = GGML_ALGO[g["quant_algo"]]
        for target in g.get("targets", []):
            out[target] = fmt
    if not out:
        raise SystemExit(f"{src}: no ggml-packed config_groups in config.json quantization_config")
    return out


def _format_of(tensor_name: str, groups: dict[str, str]) -> str | None:
    module = tensor_name.removesuffix(".weight")
    best = None
    for target, fmt in groups.items():
        if (module == target or module.startswith(target + ".")) and (best is None or len(target) > len(best[0])):
            best = (target, fmt)
    return best[1] if best else None


def export(src: Path, out: Path, llama_cpp: Path, embed_type: str = "q8_0", decoded_f32: bool = False) -> dict:
    """Convert the packed checkpoint ``src`` to ``out``; returns the export report.

    ``decoded_f32`` writes the reference instead: every tensor as the float32 decode, nothing packed. Its
    llama.cpp perplexity must match the packed GGUF's up to activation rounding (runtime equivalence check).
    """
    # llama.cpp's gguf-py must be imported before ModelOpt: transformers (pulled in by ModelOpt) imports
    # whatever gguf is installed, and the converter and its gguf-py must come from the same tree.
    gguf, get_model_architecture, get_model_class, text_type = _import_llama_cpp(llama_cpp)
    from gguf.lazy import LazyBase
    from safetensors import safe_open

    from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
    t0 = time.time()
    groups = _packed_groups(src)
    qtype_of = {f: getattr(gguf.GGMLQuantizationType, f.upper()) for f in GGML_ALGO.values()}

    # --- 1. index decoded source rows; remember where each packed tensor lives
    index = RowIndex()
    packed_files: dict[str, Path] = {}
    shard_files = sorted(src.glob("*.safetensors"))
    for shard in shard_files:
        with safe_open(str(shard), "pt") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a dict
                if f.get_slice(name).get_dtype() != "U8":
                    continue
                fmt = _format_of(name, groups)
                if fmt is None:
                    raise SystemExit(f"U8 tensor {name} matches no ggml config group")
                packed = f.get_tensor(name)
                ggml_fmt = GGML_FORMAT_REGISTRY[fmt]
                if packed.shape[-1] != ggml_fmt.block_bytes:
                    raise SystemExit(f"{name}: last dim {packed.shape[-1]} != {fmt} block bytes {ggml_fmt.block_bytes}")
                logical = torch.tensor([*packed.shape[:-2], packed.shape[-2] * ggml_fmt.block_size])
                decoded = ggml_fmt.dequantize(packed, logical, dtype=torch.float32).numpy()
                index.add(name, fmt, decoded.reshape(-1, decoded.shape[-1]))
                packed_files[name] = shard
    log.info("indexed %d packed tensors in %.1fs", len(index.sources), time.time() - t0)

    packed_cache: dict[str, np.ndarray] = {}

    def packed_rows(sid: int) -> np.ndarray:
        name = index.sources[sid][0]
        if name not in packed_cache:
            with safe_open(str(packed_files[name]), "pt") as f:
                t = f.get_tensor(name).numpy()
            packed_cache[name] = t.reshape(-1, t.shape[-2] * t.shape[-1])
        return packed_cache[name]

    # --- 2. converter subclass: decoded tensors in, packed bytes out
    hparams = json.loads((src / "config.json").read_text())
    arch = get_model_architecture(hparams, text_type)
    base_cls = get_model_class(arch)
    has_mtp = False
    for shard in shard_files:
        with safe_open(str(shard), "pt") as f:
            has_mtp = has_mtp or any(".mtp." in f".{k}" for k in f.keys())  # noqa: SIM118
    report: dict = {"src": str(src), "out": str(out), "arch": arch, "mtp_in_source": has_mtp, "tensors": {}}
    expected: dict[str, str] = {}  # gguf tensor name -> sha256 of the float32 values the converter produced

    fmt_by_bytes = {GGML_FORMAT_REGISTRY[f].block_bytes: f for f in GGML_ALGO.values()}
    if len(fmt_by_bytes) != len(GGML_ALGO):
        raise SystemExit("two GGML formats share a block size in bytes; content detection would be ambiguous")
    decoded_names: list[str] = []

    class OmkPackedModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch  # the converter's __init_subclass__ requires it in the class body
        # ForCausalLM loading drops the MTP head, so a PTQ/QAD export has none unless it was grafted back;
        # tell MTP-capable converters not to expect it (reported, never silent).
        no_mtp = (not has_mtp) or getattr(base_cls, "no_mtp", False)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hparams.pop("quantization_config", None)  # payloads are handled here, not by dequant_model

        def dequant_model(self):
            # Packed tensors are found by CONTENT (U8, last dim = one format's block bytes), never by name:
            # converters rename tensors (e.g. strip "language_model.") before this hook runs.
            for name in list(self.model_tensors):
                gen = self.model_tensors[name]
                probe = gen()
                if probe.dtype != torch.uint8 or probe.ndim < 2 or int(probe.shape[-1]) not in fmt_by_bytes:
                    continue
                ggml_fmt = GGML_FORMAT_REGISTRY[fmt_by_bytes[int(probe.shape[-1])]]

                def decode(gen=gen, ggml_fmt=ggml_fmt):
                    packed = gen()
                    if isinstance(packed, LazyBase):  # lazy mode: ModelOpt's decoder needs the real bytes
                        packed = LazyBase.to_eager(packed)
                    logical = torch.tensor([*packed.shape[:-2], packed.shape[-2] * ggml_fmt.block_size])
                    return ggml_fmt.dequantize(packed.to(torch.uint8), logical, dtype=torch.float32)

                self.model_tensors[name] = decode
                decoded_names.append(name)
            if len(decoded_names) != len(index.sources):
                raise SystemExit(
                    f"decoded {len(decoded_names)} packed tensors in the converter but indexed {len(index.sources)}"
                )

    model = OmkPackedModel(src, gguf.LlamaFileType.ALL_F32, out, eager=False)
    writer = model.gguf_writer
    real_add = writer.add_tensor

    def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        data = LazyBase.to_eager(tensor) if isinstance(tensor, LazyBase) else tensor
        if decoded_f32 or data.dtype != np.float32 or data.ndim < 2:
            real_add(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype, tensor_endianess=tensor_endianess)
            report["tensors"][name] = {"type": (raw_dtype.name if raw_dtype is not None else str(data.dtype))}
            return
        rows = data.reshape(-1, data.shape[-1])
        sid, srow = index.lookup(rows)
        digest = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
        if (sid >= 0).all():
            fmts = {index.sources[s][1] for s in np.unique(sid)}
            if len(fmts) != 1:
                raise SystemExit(f"{name}: rows come from mixed formats {sorted(fmts)}")
            fmt = fmts.pop()
            out_rows = np.empty((rows.shape[0], packed_rows(int(sid[0])).shape[1]), dtype=np.uint8)
            for s in np.unique(sid):
                sel = sid == s
                out_rows[sel] = packed_rows(int(s))[srow[sel]]
            payload = out_rows.reshape(*data.shape[:-1], -1)
            real_add(name, payload, raw_dtype=qtype_of[fmt])
            expected[name] = digest
            report["tensors"][name] = {"type": fmt.upper(), "how": "packed-bytes", "sources": len(np.unique(sid))}
            return
        if (sid >= 0).any():
            raise SystemExit(f"{name}: {(sid >= 0).sum()}/{len(sid)} rows matched packed sources (partial match)")
        is_embed = name.startswith("token_embd")
        for fmt in ("q8_0",):
            qt = qtype_of[fmt]
            if data.shape[-1] % gguf.GGML_QUANT_SIZES[qt][0]:
                continue
            q = gguf.quants.quantize(data, qt)
            exact = np.array_equal(gguf.quants.dequantize(q, qt), data)
            if exact or (is_embed and embed_type == fmt):
                real_add(name, q, raw_dtype=qt)
                if exact:
                    expected[name] = digest
                how = "exact-requant" if exact else "policy-quant"
                report["tensors"][name] = {"type": fmt.upper(), "how": how}
                return
        real_add(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype, tensor_endianess=tensor_endianess)
        report["tensors"][name] = {"type": "F32", "how": "unmatched-float"}

    writer.add_tensor = add_tensor
    model.write()
    log.info("written %s in %.1fs", out, time.time() - t0)

    # --- 5. re-read and verify every quantized tensor with gguf-py's reference decoders
    reader = gguf.GGUFReader(out)
    bad, checked = [], 0
    for t in reader.tensors:
        if t.name not in expected:
            continue
        values = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        checked += 1
        if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() != expected[t.name]:
            bad.append(t.name)
    report["verified"] = checked
    report["verify_failures"] = bad
    counts: dict[str, int] = {}
    for v in report["tensors"].values():
        counts[v["type"]] = counts.get(v["type"], 0) + 1
    report["type_counts"] = counts
    report["unmatched_float"] = [n for n, v in report["tensors"].items() if v.get("how") == "unmatched-float"]
    report["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".export.json").write_text(json.dumps(report, indent=1))
    if bad:
        raise SystemExit(f"VERIFY FAILED on {len(bad)} tensors: {bad[:5]}")
    return report


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, default=Path("/opt/llama.cpp"))
    ap.add_argument("--embed-type", default="q8_0", choices=["q8_0", "f32"])
    ap.add_argument("--decoded-f32", action="store_true", help="write the all-float32 decoded reference instead")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    r = export(a.src, a.out, a.llama_cpp, a.embed_type, a.decoded_f32)
    keys = ("arch", "mtp_in_source", "type_counts", "verified", "verify_failures", "unmatched_float", "seconds")
    print(json.dumps({k: r[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
