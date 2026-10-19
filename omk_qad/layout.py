# SPDX-License-Identifier: Apache-2.0
"""Per-tensor GGUF layout of an HF causal LM: names, source rows/columns and the llama.cpp quant plan.

Two questions, both answered by llama.cpp itself rather than re-implemented here:

  1. Which GGUF tensor holds which HF values?  llama.cpp's converter renames, splits fused tensors and
     permutes heads. The converter is run on synthetic index tensors (no weights are read): one pass where
     every value is its source tensor's id, one with its source row, one with its source column. For each
     GGUF tensor that gives, per GGUF row, the source parameter and row, plus the column order. A GGML
     block covers 256 consecutive values of one row, so (rows, column order) is exactly what the trainer
     needs to fake-quantize the SAME blocks the GGUF will hold.
  2. Which type does each GGUF tensor get?  A shape-only sparse GGUF (header + tensor info, no data on
     disk) is written and ``llama-quantize --dry-run <ftype>`` prints its per-tensor plan. Classes trained
     at a fixed format (e.g. experts at IQ2_XXS) are passed as ``--tensor-type`` overrides, so llama.cpp's
     own counters (first n/8 ffn_down, ...) behave as in a real llama-quantize run with those overrides.

  python -m omk_qad.layout --model DIR --out LAYOUT.pt [--ftype IQ2_M]
                           [--fixed routed_experts=iq2_xxs,shared_experts=iq2_xxs,dense_mlp=iq2_xxs]

DIR needs config.json and the tokenizer files only. LAYOUT.pt (torch) holds the index arrays; a JSON
summary (types, sizes, per-class formats, unsupported tensors) is written beside it.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .gguf_export import _import_llama_cpp

log = logging.getLogger("omk_qad.layout")

FLOAT_TYPES = {"F32", "F16", "BF16"}
DEFAULT_FIXED = {"routed_experts": "iq2_xxs", "shared_experts": "iq2_xxs", "dense_mlp": "iq2_xxs"}
_PLAN_LINE = re.compile(
    r"^\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[([^\]]*)\], type =\s*(\S+), size =.*?(?:->\s*[\d.]+ MiB \((\S+)\))?\s*$"
)
_SIZE_LINE = re.compile(r"(model|quant) size\s*=\s*([\d.]+) MiB")


def meta_text_model(model_dir: Path):
    """The text causal LM of ``model_dir`` on the meta device (no weights read) and its text config."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir).get_text_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return model, config


def param_classes(model) -> dict[str, str]:
    """Census class of every parameter (exact entry, else the longest module prefix, else 'vectors')."""
    from .census import classify_model

    census = classify_model(model)
    exact = {e.module: e.cls for e in census.entries}
    out = {}
    for name, p in model.named_parameters():
        if name in exact:
            out[name] = exact[name]
            continue
        best = None
        for mod, cls in exact.items():
            if name.startswith(mod + ".") and (best is None or len(mod) > len(best[0])):
                best = (mod, cls)
        out[name] = best[1] if best else ("vectors" if p.ndim <= 1 else "other")
    return out


def _rows_cols(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    return int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1, int(shape[-1])


def _probe(shape: tuple[int, ...], kind: str, src_id: int) -> torch.Tensor:
    rows, cols = _rows_cols(shape)
    if kind == "src":
        t = torch.full((rows, cols), float(src_id))
    elif kind == "row":
        t = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, cols)
    else:
        t = torch.arange(cols, dtype=torch.float32)[None, :].expand(rows, cols)
    return t.reshape(shape).contiguous()


def _converter_pass(
    model_dir: Path, llama_cpp: Path, params: dict[str, tuple[int, ...]], kind: str, sink, sparse_out=None
):
    """Run llama.cpp's converter for ``model_dir``'s arch on index probes of ``kind``; ``sink(name, data)``."""
    gguf, get_model_architecture, get_model_class, text_type = _import_llama_cpp(llama_cpp)
    hparams = json.loads((model_dir / "config.json").read_text())
    base_cls = get_model_class(get_model_architecture(hparams, text_type))
    names = list(params)

    class ProbeModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch
        no_mtp = True  # the text causal LM has no MTP head; see gguf_export

        def index_tensors(self, remote_hf_model_id=None):
            # Only the base, file-reading ModelBase.index_tensors is replaced, for this call: arch subclasses
            # wrap it with their own setup (e.g. Qwen3.5 records the block count that filter_tensors needs).
            from conversion.base import ModelBase

            def probes(model_self, remote_hf_model_id=None):
                del remote_hf_model_id
                out = {}
                for sid, name in enumerate(names):
                    gen = lambda name=name, sid=sid: _probe(params[name], kind, sid)  # noqa: E731
                    item = model_self.filter_tensors((name, gen))
                    if item:
                        out[item[0]] = item[1]
                return out

            real = ModelBase.index_tensors
            ModelBase.index_tensors = probes
            try:
                return super().index_tensors(remote_hf_model_id=remote_hf_model_id)
            finally:
                ModelBase.index_tensors = real

    model = ProbeModel(model_dir, gguf.LlamaFileType.ALL_F32, sparse_out or Path("/dev/null"), eager=True)
    writer = model.gguf_writer

    def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        del raw_shape, tensor_endianess
        data = np.asarray(tensor)
        if raw_dtype is not None and raw_dtype != gguf.GGMLQuantizationType.F32:
            raise SystemExit(f"{name}: converter chose {raw_dtype.name} under ALL_F32; layout expects F32")
        if sparse_out is not None:
            writer.add_tensor_info(name, data.shape, np.dtype(np.float32), data.size * 4)
        sink(name, data)

    writer.add_tensor = add_tensor
    model.prepare_tensors()
    if sparse_out is None:
        return
    # shape-only GGUF: header, metadata and tensor info, then a sparse data region of the right size
    model.prepare_metadata(vocab_only=False)
    writer.write_header_to_file(path=sparse_out)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    fout = writer.fout[0]
    align = writer.data_alignment
    start = writer.ggml_pad(fout.tell(), align)
    total = start + sum(writer.ggml_pad(ti.nbytes, align) for ti in writer.tensors[0].values())
    fout.truncate(total)
    writer.close()


def _dry_run(llama_cpp: Path, sparse: Path, ftype: str, overrides: dict[str, str], work: Path) -> tuple[dict, dict]:
    """llama-quantize --dry-run plan: {gguf name: target type (upper)}, {"model": MiB, "quant": MiB}."""
    cmd = [str(llama_cpp / "build" / "bin" / "llama-quantize"), "--dry-run"]
    if overrides:
        tt = work / f"{sparse.stem}.tensor_types.txt"
        tt.write_text("".join(f"^{re.escape(n)}$={t}\n" for n, t in sorted(overrides.items())))
        cmd += ["--tensor-type-file", str(tt)]
    cmd += [str(sparse), ftype]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    text = proc.stdout + proc.stderr
    (work / f"{sparse.stem}.dry_run.log").write_text(text)
    if proc.returncode != 0:
        raise SystemExit(f"llama-quantize --dry-run failed ({proc.returncode}): {work}/{sparse.stem}.dry_run.log")
    plan, sizes = {}, {}
    for line in text.splitlines():
        m = _PLAN_LINE.match(line.strip())
        if m:
            plan[m.group(1)] = (m.group(4) or m.group(3)).upper()
        s = _SIZE_LINE.search(line)
        if s:
            sizes[s.group(1)] = float(s.group(2))
    return plan, sizes


def build_layout(model_dir: Path, llama_cpp: Path, ftype: str = "IQ2_M", fixed: dict[str, str] | None = None,
                 work: Path | None = None) -> dict:
    """Probe the converter and ask llama-quantize for the plan; returns the layout dict (see module doc)."""
    t0 = time.time()
    fixed = DEFAULT_FIXED if fixed is None else fixed
    model_dir = Path(model_dir)
    work = Path(work or model_dir)
    work.mkdir(parents=True, exist_ok=True)
    model, config = meta_text_model(model_dir)
    params = {n: tuple(p.shape) for n, p in model.named_parameters()}
    classes = param_classes(model)
    names = list(params)
    del model

    # pass 1 (source ids) also writes the sparse shape-only GGUF; passes 2/3 give rows and columns
    src: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    bad: dict[str, str] = {}

    def sink_src(name, data):
        shapes[name] = tuple(data.shape)
        rows, cols = _rows_cols(data.shape)
        d = data.reshape(rows, cols)
        first = d[:, 0]
        if not (np.array_equal(d, np.broadcast_to(first[:, None], d.shape)) and np.all(first == np.round(first))
                and np.all((first >= 0) & (first < len(names)))):
            bad[name] = "values are not a per-row copy of one source (mixed or transformed)"
            return
        src[name] = first.astype(np.int64)

    sparse = work / "layout_shapes.gguf"
    _converter_pass(model_dir, llama_cpp, params, "src", sink_src, sparse_out=sparse)

    rows_of: dict[str, np.ndarray] = {}

    def sink_row(name, data):
        if name in bad:
            return
        rows, cols = _rows_cols(data.shape)
        d = data.reshape(rows, cols)
        first = d[:, 0]
        if not (np.array_equal(d, np.broadcast_to(first[:, None], d.shape)) and np.all(first == np.round(first))):
            bad[name] = "a GGUF row mixes source rows (transpose or reshape across rows)"
            return
        rows_of[name] = first.astype(np.int64)

    _converter_pass(model_dir, llama_cpp, params, "row", sink_row)

    cols_of: dict[str, np.ndarray | None] = {}

    def sink_col(name, data):
        if name in bad:
            return
        rows, cols = _rows_cols(data.shape)
        d = data.reshape(rows, cols)
        first = d[0]
        if not np.array_equal(d, np.broadcast_to(first[None, :], d.shape)):
            bad[name] = "column order differs between rows"
            return
        p = names[int(src[name][0])]
        width = _rows_cols(params[p])[1]
        if len({names[int(s)] for s in np.unique(src[name])}) != 1:
            bad[name] = "rows come from several source tensors"
            return
        if width != cols or not np.array_equal(np.sort(first), np.arange(cols)):
            bad[name] = f"columns are not a permutation of the source's {width} columns"
            return
        cols_of[name] = None if np.array_equal(first, np.arange(cols)) else first.astype(np.int64)

    _converter_pass(model_dir, llama_cpp, params, "col", sink_col)

    # overrides for fixed-format classes, by exact GGUF name
    def source_of(g):
        return names[int(src[g][0])] if g in src else None

    overrides = {}
    for g in shapes:
        p = source_of(g)
        if p is not None and g not in bad and classes.get(p) in fixed and len(shapes[g]) >= 2:
            overrides[g] = fixed[classes[p]]
    plan, sizes = _dry_run(llama_cpp, sparse, ftype, overrides, work)
    missing = sorted(set(shapes) - set(plan))
    if missing:
        raise SystemExit(f"llama-quantize plan lacks {len(missing)} tensors, e.g. {missing[:5]}")

    targets = {}
    for g, shape in shapes.items():
        # a transformed tensor (e.g. norm + 1) shifts the source-id probe too, so its source is not trusted
        p = source_of(g) if g in cols_of else None
        targets[g] = {
            "shape": list(shape),
            "type": plan[g],
            "src": p,
            "cls": classes.get(p) if p else None,
            "rows": torch.from_numpy(rows_of[g]) if g in rows_of else None,
            "colperm": torch.from_numpy(cols_of[g]) if cols_of.get(g) is not None else None,
            "mapping": "direct" if g in cols_of else f"unsupported: {bad.get(g, 'unknown')}",
        }
    by_param = defaultdict(list)
    for g, t in targets.items():
        if t["src"] is not None and t["mapping"] == "direct":
            by_param[t["src"]].append(g)
    layout = {
        "model_dir": str(model_dir),
        "ftype": ftype,
        "fixed": fixed,
        "llama_cpp": str(llama_cpp),
        "params": {n: list(s) for n, s in params.items()},
        "classes": classes,
        "targets": targets,
        "by_param": dict(by_param),
        "sizes_mib": sizes,
        "seconds": round(time.time() - t0, 1),
    }
    sparse.unlink(missing_ok=True)
    return layout


def summary(layout: dict) -> dict:
    """JSON-able summary: type counts, per-class types, unsupported tensors, plan sizes."""
    per_class: dict[str, Counter] = defaultdict(Counter)
    for t in layout["targets"].values():
        per_class[t["cls"] or "-"][t["type"]] += 1
    return {
        "model_dir": layout["model_dir"],
        "ftype": layout["ftype"],
        "fixed": layout["fixed"],
        "type_counts": dict(Counter(t["type"] for t in layout["targets"].values())),
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "unsupported": {g: t["mapping"] for g, t in layout["targets"].items() if t["mapping"] != "direct"},
        "unsupported_quantized": sorted(
            g for g, t in layout["targets"].items() if t["mapping"] != "direct" and t["type"] not in FLOAT_TYPES
        ),
        "permuted_columns": sorted(g for g, t in layout["targets"].items() if t["colperm"] is not None),
        "sizes_mib": layout["sizes_mib"],
        "seconds": layout["seconds"],
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, default=Path("/opt/llama.cpp"))
    ap.add_argument("--ftype", default="IQ2_M")
    ap.add_argument("--fixed", default=",".join(f"{k}={v}" for k, v in DEFAULT_FIXED.items()),
                    help="class=type pairs forced before llama.cpp plans the rest ('' = pure llama.cpp mix)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fixed = dict(kv.split("=", 1) for kv in a.fixed.split(",") if kv)
    layout = build_layout(a.model, a.llama_cpp, a.ftype, fixed, work=a.out.parent)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(layout, a.out)
    s = summary(layout)
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("type_counts", "per_class", "sizes_mib", "seconds")}, indent=1))
    print(f"unsupported mappings: {len(s['unsupported'])} (quantized among them: {s['unsupported_quantized']})")
    print(f"permuted columns: {s['permuted_columns']}")


if __name__ == "__main__":
    main()
