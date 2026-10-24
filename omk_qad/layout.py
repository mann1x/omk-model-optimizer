# SPDX-License-Identifier: Apache-2.0
r"""Per-tensor GGUF layout of an HF causal LM: names, source rows/columns and the llama.cpp quant plan.

Two questions, both answered by llama.cpp itself rather than re-implemented here:

  1. Which GGUF tensor holds which HF values?  llama.cpp's converter renames, splits fused tensors and
     permutes heads. The converter is run on synthetic index tensors (no weights are read): one pass where
     every value is its source tensor's id, one with its source row, one with its source column. For each
     GGUF tensor that gives, per GGUF row, the source parameter and row, plus the column order. A GGML
     block covers 256 consecutive values of one row, so (rows, column order) is exactly what the trainer
     needs to fake-quantize the SAME blocks the GGUF will hold.
  2. Which type does each GGUF tensor get?  A shape-only sparse GGUF (header + tensor info, no data on
     disk) is written and ``llama-quantize --dry-run <ftype>`` prints its per-tensor plan. Classes given a
     fixed format (e.g. experts at IQ2_XXS) are passed as ``--tensor-type`` overrides, so llama.cpp's
     own counters (first n/8 ffn_down, ...) behave as in a real llama-quantize run with those overrides.

  python -m omk_qad.layout --model DIR --out LAYOUT.pt [--rest IQ2_M|IQ2_S|IQ3_XXS|Q2_K|...|bolt]
                           [--fixed routed_experts=iq2_xxs,shared_experts=iq2_xxs,dense_mlp=iq2_xxs]
                           [--override linear_attn=q4_k] [--override-regex 'attn_v\.weight$=q6_k']
                           [--compare IQ2_M,IQ2_S,IQ3_XXS,Q2_K,bolt]

At least one fixed class is mandatory: there is no pure llama.cpp-mix mode. "Fixed" only means the type is set
by hand instead of by the --rest ftype's rules. It does not decide what trains: omk_qad.train fake-quantizes and
trains EVERY quantized tensor at its planned type, fixed or rest, except the classes in its --export-only.

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
# The rest of the model follows a llama.cpp ftype's own rules; "bolt" = NVIDIA Bolt's map (Q8_0 for everything
# not fixed; Bolt's NVFP4 Mamba projections have no verified llama.cpp path here).
REST_PRESETS = ("IQ2_M", "IQ2_S", "IQ2_XS", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M",
                "Q4_K_S", "Q4_K_M", "bolt")
REST_ALIASES = {"bolt": "Q8_0"}
# MTP (nextn) head: Q4_K like our published GGUFs' MTP-head override (an IQ2 head has no imatrix coverage and
# drafts poorly); "rest" lets llama.cpp's own rules decide.
DEFAULT_MTP_TYPE = "Q4_K"
DEFAULT_FIXED = {"routed_experts": "iq2_xxs", "shared_experts": "iq2_xxs", "dense_mlp": "iq2_xxs"}
_PLAN_LINE = re.compile(
    r"^\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[([^\]]*)\], type =\s*(\S+), size =.*?(?:->\s*[\d.]+ MiB \((\S+)\))?\s*$"
)
_SIZE_LINE = re.compile(r"(model|quant) size\s*=\s*([\d.]+) MiB")


def meta_text_model(model_dir: Path, mtp: bool = True):
    """The text causal LM of ``model_dir`` on the meta device (no weights read), with its MTP head if declared."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .mtp import attach, mtp_layers

    config = AutoConfig.from_pretrained(model_dir).get_text_config()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
        if mtp and mtp_layers(config):
            attach(model)
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
    for name in out:  # the MTP head is its own class (its own format), whatever its structure
        if name.startswith("mtp."):
            out[name] = "mtp"
    emb_in, emb_out = model.get_input_embeddings(), model.get_output_embeddings()
    if emb_out is not None and emb_in.weight is emb_out.weight:
        # tied: the one tensor IS the head (llama.cpp ships it as token_embd with the output's rules); classing it
        # "embedding" would make it export-only, so the head would train unquantized and ship quantized
        tied = next(n for n, p in model.named_parameters() if p is emb_in.weight)
        out[tied] = "lm_head"
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
    model_dir: Path, llama_cpp: Path, params: dict[str, tuple[int, ...]], kind: str, sink, sparse_out=None,
    mtp: bool = False,
):
    """Run llama.cpp's converter for ``model_dir``'s arch on index probes of ``kind``; ``sink(name, data)``."""
    gguf, get_model_architecture, get_model_class, text_type = _import_llama_cpp(llama_cpp)
    hparams = json.loads((model_dir / "config.json").read_text())
    base_cls = get_model_class(get_model_architecture(hparams, text_type))
    names = list(params)

    class ProbeModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch
        no_mtp = not mtp  # MTP tensors present -> the converter maps them to blk.<n_layer>.nextn.*

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


def _dry_run(
    llama_cpp: Path, sparse: Path, ftype: str, patterns: list[tuple[str, str]], work: Path
) -> tuple[dict, dict]:
    """llama-quantize --dry-run plan: {gguf name: target type (upper)}, {"model": MiB, "quant": MiB}."""
    cmd = [str(llama_cpp / "build" / "bin" / "llama-quantize"), "--dry-run"]
    if patterns:  # llama.cpp applies the FIRST matching pattern, so order is precedence
        tt = work / f"{sparse.stem}.{ftype}.tensor_types.txt"
        tt.write_text("".join(f"{pat}={t.lower()}\n" for pat, t in patterns))
        cmd += ["--tensor-type-file", str(tt)]
    cmd += [str(sparse), ftype]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    text = proc.stdout + proc.stderr
    (work / f"{sparse.stem}.{ftype}.dry_run.log").write_text(text)
    if proc.returncode != 0:
        log_file = work / f"{sparse.stem}.{ftype}.dry_run.log"
        raise SystemExit(f"llama-quantize --dry-run {ftype} failed ({proc.returncode}): {log_file}")
    plan, sizes = {}, {}
    for line in text.splitlines():
        m = _PLAN_LINE.match(line.strip())
        if m:
            plan[m.group(1)] = (m.group(4) or m.group(3)).upper()
        s = _SIZE_LINE.search(line)
        if s:
            sizes[s.group(1)] = float(s.group(2))
    return plan, sizes


def probe_model(model_dir: Path, llama_cpp: Path, work: Path, mtp: bool = True) -> dict:
    """Converter index probes (no weights read): per GGUF tensor its shape, source rows and column order.

    Also writes the sparse shape-only GGUF every plan is computed from (``probe["sparse"]``).
    """
    t0 = time.time()
    model_dir, work = Path(model_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    model, _ = meta_text_model(model_dir, mtp)
    params = {n: tuple(p.shape) for n, p in model.named_parameters()}
    classes = param_classes(model)
    names = list(params)
    del model
    has_mtp = any(n.startswith("mtp.") for n in names)

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
    _converter_pass(model_dir, llama_cpp, params, "src", sink_src, sparse_out=sparse, mtp=has_mtp)

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

    _converter_pass(model_dir, llama_cpp, params, "row", sink_row, mtp=has_mtp)

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

    _converter_pass(model_dir, llama_cpp, params, "col", sink_col, mtp=has_mtp)
    # a transformed tensor (e.g. norm + 1) shifts the source-id probe too, so only proven mappings keep a source
    source = {g: names[int(src[g][0])] for g in cols_of}
    return {"model_dir": str(model_dir), "llama_cpp": str(llama_cpp), "params": params, "classes": classes,
            "shapes": shapes, "source": source, "rows": rows_of, "cols": cols_of, "bad": bad, "sparse": sparse,
            "work": work, "has_mtp": has_mtp, "seconds": round(time.time() - t0, 1)}


def _check_type(t: str, llama_cpp: Path) -> str:
    from .ternary import TYPES as TERNARY

    if t.upper() in TERNARY:
        return t.upper()
    gguf = _import_llama_cpp(Path(llama_cpp))[0]  # gguf-py must be llama.cpp's, never a pip gguf
    name = t.upper()
    if name not in gguf.GGMLQuantizationType.__members__:
        raise SystemExit(f"unknown GGML type {t!r}")
    return name


TERNARY_TYPES = ("PQ2_0", "PTQ1_0")


def _check_ternary(tern: dict[str, str], shapes: dict, sizes: dict | None = None) -> int | None:
    """Validate ternary targets against the runtime contract; correct the plan size; return the model's block."""
    from .ternary import BLOCK_BYTES, G, foldable, model_block

    if not tern:
        return None
    bad = sorted(g for g in tern if not foldable(g))
    if bad:
        raise SystemExit(f"the runtime does not fold {bad[:5]}: ternary types apply only to attention / linear-"
                         "attention ssm_out / FFN and expert weights / output / token_embd (omk_qad.ternary.foldable)")
    odd = sorted(g for g in tern if len(shapes[g]) < 2 or shapes[g][-1] % G)
    if odd:
        raise SystemExit(f"ternary needs 2-D+ weights with an input width a multiple of {G}: {odd[:5]}")
    if sizes is not None:  # the dry run counted IQ2_XXS (66 B / 256) for these tensors
        delta = sum(int(np.prod(shapes[g])) * (BLOCK_BYTES[t] / G - 66 / 256) for g, t in tern.items())
        sizes["quant"] = round(sizes["quant"] + delta / 2**20, 2)
    return model_block(shapes[g][-1] for g in tern)


def retype(layout: dict, cls_types: dict[str, str]) -> dict:
    """A copy of ``layout`` with every quantized target of the given classes at a new type.

    E.g. a ternary student from an IQ2 plan; validated as :func:`plan_layout` does; sizes are not recomputed.
    """
    import copy

    out = copy.deepcopy(layout)
    cls_types = {c: _check_type(t, Path(layout["llama_cpp"])) for c, t in cls_types.items()}
    for t in out["targets"].values():
        if t["cls"] in cls_types and t["type"] not in FLOAT_TYPES:
            t["type"] = cls_types[t["cls"]]
    tern = {g: t["type"] for g, t in out["targets"].items() if t["type"] in TERNARY_TYPES}
    out["hadamard_block"] = _check_ternary(tern, {g: t["shape"] for g, t in out["targets"].items()})
    out["fixed"] = {**out["fixed"], **{c: t for c, t in cls_types.items() if c in out["fixed"]}}
    return out


def plan_layout(probe: dict, rest: str = "IQ2_M", fixed: dict[str, str] | None = None,
                overrides: dict[str, str] | None = None, regex: dict[str, str] | None = None,
                mtp_type: str = DEFAULT_MTP_TYPE) -> dict:
    """The layout for one preset: ``fixed`` classes at their trained format, everything else per llama.cpp.

    ``rest`` is the llama.cpp ftype whose rules plan the non-fixed tensors (or ``bolt`` = Q8_0 rest);
    ``overrides`` force a type per census class, ``regex`` per GGUF-name pattern. Precedence (llama.cpp applies
    the first matching pattern): fixed > class overrides > regex overrides > ``rest`` rules.
    """
    t0 = time.time()
    fixed = DEFAULT_FIXED if fixed is None else fixed
    if not fixed:
        raise SystemExit("no fixed class: a pure llama.cpp mix is not a QAD layout (set --fixed)")
    rest_ftype = REST_ALIASES.get(rest.lower(), rest.upper())
    lc = probe["llama_cpp"]
    fixed = {c: _check_type(t, lc) for c, t in fixed.items()}
    overrides = {c: _check_type(t, lc) for c, t in (overrides or {}).items()}
    regex = {r: _check_type(t, lc) for r, t in (regex or {}).items()}
    if probe.get("has_mtp") and mtp_type.lower() != "rest":  # "rest": llama.cpp's own rules for the nextn block
        overrides = {"mtp": _check_type(mtp_type, lc), **overrides}
    if set(fixed) & set(overrides):
        raise SystemExit(f"classes both fixed and overridden: {sorted(set(fixed) & set(overrides))}")
    shapes, source, classes = probe["shapes"], probe["source"], probe["classes"]
    exact: dict[str, str] = {}
    for table in (fixed, overrides):
        for g in shapes:
            p = source.get(g)
            if p is not None and classes.get(p) in table and len(shapes[g]) >= 2 and g not in exact:
                exact[g] = table[classes[p]]
    if not any(classes.get(source.get(g)) in fixed for g in exact):
        raise SystemExit(f"no tensor of the fixed classes {sorted(fixed)} exists in this model")
    if any(t in TERNARY_TYPES for t in regex.values()):
        raise SystemExit("ternary types are set per class (--fixed / --override), not by --override-regex")
    tern = {g: t for g, t in exact.items() if t in TERNARY_TYPES}
    # llama-quantize does not know the ternary types: plan them as an IQ2_XXS placeholder, then put them back
    patterns = [(f"^{re.escape(g)}$", "IQ2_XXS" if t in TERNARY_TYPES else t) for g, t in exact.items()]
    patterns += list(regex.items())
    plan, sizes = _dry_run(Path(probe["llama_cpp"]), probe["sparse"], rest_ftype, patterns, Path(probe["work"]))
    plan.update(tern)
    block = _check_ternary(tern, shapes, sizes)
    missing = sorted(set(shapes) - set(plan))
    if missing:
        raise SystemExit(f"llama-quantize plan lacks {len(missing)} tensors, e.g. {missing[:5]}")

    targets = {}
    for g, shape in shapes.items():
        p = source.get(g)
        targets[g] = {
            "shape": list(shape),
            "type": plan[g],
            "src": p,
            "cls": classes.get(p) if p else None,
            "rows": torch.from_numpy(probe["rows"][g]) if p else None,
            "colperm": torch.from_numpy(probe["cols"][g]) if p and probe["cols"][g] is not None else None,
            "mapping": "direct" if p else f"unsupported: {probe['bad'].get(g, 'unknown')}",
        }
    by_param = defaultdict(list)
    for g, t in targets.items():
        if t["src"] is not None:
            by_param[t["src"]].append(g)
    return {
        "model_dir": probe["model_dir"],
        "ftype": rest_ftype,
        "rest": rest,
        "fixed": fixed,
        "overrides": overrides,
        "regex": regex,
        "mtp": bool(probe.get("has_mtp")),
        "mtp_type": mtp_type if probe.get("has_mtp") else None,
        "llama_cpp": probe["llama_cpp"],
        "params": {n: list(s) for n, s in probe["params"].items()},
        "classes": classes,
        "targets": targets,
        "by_param": dict(by_param),
        "sizes_mib": sizes,
        "hadamard_block": block,
        "seconds": round(probe["seconds"] + time.time() - t0, 1),
    }


def build_layout(model_dir: Path, llama_cpp: Path, rest: str = "IQ2_M", fixed: dict[str, str] | None = None,
                 work: Path | None = None, overrides: dict[str, str] | None = None,
                 regex: dict[str, str] | None = None, mtp: bool = True, mtp_type: str = DEFAULT_MTP_TYPE) -> dict:
    """Probe + plan in one call (see :func:`probe_model`, :func:`plan_layout`)."""
    probe = probe_model(model_dir, llama_cpp, Path(work or model_dir), mtp)
    try:
        return plan_layout(probe, rest, fixed, overrides, regex, mtp_type)
    finally:
        probe["sparse"].unlink(missing_ok=True)


def summary(layout: dict) -> dict:
    """JSON-able summary: type counts, per-class types, unsupported tensors, plan sizes."""
    per_class: dict[str, Counter] = defaultdict(Counter)
    for t in layout["targets"].values():
        per_class[t["cls"] or "-"][t["type"]] += 1
    return {
        "model_dir": layout["model_dir"],
        "ftype": layout["ftype"],
        "rest": layout.get("rest", layout["ftype"]),
        "fixed": layout["fixed"],
        "overrides": layout.get("overrides", {}),
        "mtp": layout.get("mtp", False),
        "mtp_type": layout.get("mtp_type"),
        "regex": layout.get("regex", {}),
        "type_counts": dict(Counter(t["type"] for t in layout["targets"].values())),
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "unsupported": {g: t["mapping"] for g, t in layout["targets"].items() if t["mapping"] != "direct"},
        "unsupported_quantized": sorted(
            g for g, t in layout["targets"].items() if t["mapping"] != "direct" and t["type"] not in FLOAT_TYPES
        ),
        "permuted_columns": sorted(g for g, t in layout["targets"].items() if t["colperm"] is not None),
        "sizes_mib": layout["sizes_mib"],
        "hadamard_block": layout.get("hadamard_block"),
        "seconds": layout["seconds"],
    }


def _pairs(text: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in text.split(",") if kv)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, default=Path("/opt/llama.cpp"))
    ap.add_argument("--rest", "--ftype", dest="rest", default="IQ2_M",
                    help=f"llama.cpp ftype whose rules plan the non-fixed tensors: {', '.join(REST_PRESETS)}")
    ap.add_argument("--fixed", default=",".join(f"{k}={v}" for k, v in DEFAULT_FIXED.items()),
                    help="class=type pairs trained at a fixed format (required, non-empty)")
    ap.add_argument("--override", default="", help="class=type pairs for non-fixed classes, e.g. linear_attn=q4_k")
    ap.add_argument("--override-regex", action="append", default=[],
                    help="GGUF-name regex=type (repeatable), e.g. 'attn_v\\.weight$=q6_k'")
    ap.add_argument("--no-mtp", action="store_true", help="leave the MTP (nextn) head out even if declared")
    ap.add_argument("--mtp-type", default=DEFAULT_MTP_TYPE, help="type of the MTP head tensors, or 'rest'")
    ap.add_argument("--compare", default="", help="comma list of rest presets: print a size table, save --rest only")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fixed, overrides = _pairs(a.fixed), _pairs(a.override)
    regex = dict(r.rsplit("=", 1) for r in a.override_regex)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    probe = probe_model(a.model, a.llama_cpp, a.out.parent, not a.no_mtp)
    try:
        if a.compare:
            rows = []
            for rest in a.compare.split(","):
                lay = plan_layout(probe, rest, fixed, overrides, regex, a.mtp_type)
                gb = lay["sizes_mib"].get("quant", float("nan")) * 2**20 / 1e9
                types = Counter(t["type"] for t in lay["targets"].values() if t["cls"] not in fixed)
                rows.append(f"{rest:8s} {gb:7.2f} GB   rest types: {dict(types.most_common())}")
            print("\n".join(rows))
        layout = plan_layout(probe, a.rest, fixed, overrides, regex, a.mtp_type)
    finally:
        probe["sparse"].unlink(missing_ok=True)
    if a.out.exists():  # a layout may be in use by a run (its identity names this file): never replaced by another
        old = torch.load(a.out, weights_only=False)
        types = [{g: t["type"] for g, t in x["targets"].items()} for x in (old, layout)]
        same = types[0] == types[1] and all(old.get(k) == layout.get(k) for k in ("params", "hadamard_block"))
        if not same:
            raise SystemExit(f"RUN_REFUSED layout: {a.out} exists with another plan; write the new one elsewhere")
        print(f"{a.out}: the same plan already exists, kept as it is")
    else:
        from .runctl import atomic_json, atomic_torch_save

        atomic_json(a.out.with_suffix(".json"), summary(layout))
        atomic_torch_save(layout, a.out)  # last: the .pt is what runs load
    s = summary(layout)
    keys = ("rest", "fixed", "type_counts", "per_class", "sizes_mib", "seconds")
    print(json.dumps({k: s[k] for k in keys}, indent=1))
    print(f"unsupported mappings: {len(s['unsupported'])} (quantized among them: {s['unsupported_quantized']})")
    print(f"permuted columns: {s['permuted_columns']}")


if __name__ == "__main__":
    main()
