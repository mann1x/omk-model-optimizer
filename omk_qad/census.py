# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic tensor census and ModelOpt quant config for 2-bit QAD.

Every weight of an HF causal LM is put in one class by STRUCTURE, not by a per-architecture name table:

  routed_experts  a module holding 3-D expert parameters (fused HF experts: gate_up_proj / up_proj + down_proj),
                  or nn.Linear under a ModuleList of experts (per-expert checkpoints)
  router          nn.Linear sibling of an experts module whose out_features == num_experts, plus tiny gates
  shared_experts  nn.Linear under a module named shared_expert(s) inside a MoE block
  dense_mlp       nn.Linear under a module whose class name ends in "MLP" (dense FFN)
  attention       nn.Linear under a module whose class name contains "Attention" (softmax attention)
  linear_attn     nn.Linear under GatedDeltaNet / linear-attention / Mamba-style mixers
  small_proj      any nn.Linear with out_features < SMALL_OUT (gated-delta a/b, shared_expert_gate, ...)
  embedding       model.get_input_embeddings()
  lm_head         model.get_output_embeddings()
  other           anything else; reported loudly, never silently quantized

A policy maps class -> GGML format. A format is only applied where the input dim is a multiple of its block
size; otherwise the census falls back down FALLBACK and says so. Nothing is decided silently.

The returned quant_cfg names exact module paths (no wildcards over names), so it works on any model whose
structure the classifier recognises, and the census proves which modules it reached before any run.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["DEFAULT_POLICY", "FORMATS", "Census", "build_quant_cfg", "classify_model", "run_census"]

SMALL_OUT = 128
FORMATS = {
    # name: (block size along the input dim, bits per weight)
    "iq1_s": (256, 1.5625),
    "iq1_m": (256, 1.75),
    "iq2_xxs": (256, 2.0625),
    "iq2_xs": (256, 2.3125),
    "iq2_s": (256, 2.5625),
    "q8_0": (32, 8.5),
    "none": (1, 16.0),
}
FALLBACK = {"iq1_s": "q8_0", "iq1_m": "q8_0", "iq2_xxs": "q8_0", "iq2_xs": "q8_0", "iq2_s": "q8_0", "q8_0": "none"}
# Bolt's map (report Table 2), with the Mamba-style NVFP4 layers held at Q8_0 until llama.cpp NVFP4 is verified here.
DEFAULT_POLICY = {
    "routed_experts": "iq2_xxs",
    "shared_experts": "iq2_xxs",
    "dense_mlp": "iq2_xxs",
    "attention": "q8_0",
    "linear_attn": "q8_0",
    "small_proj": "none",
    "router": "none",
    "embedding": "q8_0",  # applied at GGUF export (ModelOpt does not fake-quant nn.Embedding)
    "lm_head": "q8_0",
    "other": "none",
    "vectors": "none",
}
EXPORT_ONLY = {"embedding"}
LINEAR_ATTN_HINTS = ("gateddelta", "linearattention", "linear_attn", "mamba", "mixer", "ssm")


@dataclass
class Entry:
    """One weight-bearing module."""

    module: str
    cls: str
    kind: str  # "linear" | "fused_experts" | "embedding"
    params: int
    in_dim: int
    fmt: str = "none"
    note: str = ""


@dataclass
class Census:
    """Classification of every weight-bearing module plus the format chosen for it."""

    entries: list[Entry] = field(default_factory=list)
    model_params: int = 0

    def by_class(self) -> dict[str, list[Entry]]:
        """Entries grouped by class."""
        out: dict[str, list[Entry]] = defaultdict(list)
        for e in self.entries:
            out[e.cls].append(e)
        return dict(out)

    def size_bytes(self) -> float:
        """Estimated stored size of the classified weights at their chosen formats."""
        return sum(e.params * FORMATS[e.fmt][1] / 8 for e in self.entries)

    def table(self) -> str:
        """Human-readable census table."""
        rows = [f"{'class':15s} {'modules':>8s} {'params':>14s} {'format(s)':24s} {'GB':>8s}"]
        total_p = 0
        for cls, es in sorted(self.by_class().items()):
            p = sum(e.params for e in es)
            total_p += p
            fmts = ",".join(sorted({e.fmt for e in es}))
            gb = sum(e.params * FORMATS[e.fmt][1] / 8 for e in es) / 1e9
            rows.append(f"{cls:15s} {len(es):8d} {p:14,d} {fmts:24s} {gb:8.3f}")
        rows.append(f"{'TOTAL':15s} {len(self.entries):8d} {total_p:14,d} {'':24s} {self.size_bytes() / 1e9:8.3f}")
        if self.model_params:
            status = "OK" if total_p == self.model_params else "MISMATCH"
            rows.append(f"coverage: census {total_p:,d} / model {self.model_params:,d} params -> {status}")
        notes = [f"  ! {e.module}: {e.note}" for e in self.entries if e.note]
        return "\n".join(rows + (["fallbacks / notes:", *notes] if notes else []))

    def to_json(self) -> dict:
        """Serialisable form (written next to every run)."""
        return {
            "entries": [asdict(e) for e in self.entries],
            "size_gb": self.size_bytes() / 1e9,
            "model_params": self.model_params,
        }


def _expert_tensors(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(first projection, down projection) of a fused HF experts module, or None."""
    down = getattr(module, "down_proj", None)
    first = getattr(module, "gate_up_proj", None)
    if first is None:
        first = getattr(module, "up_proj", None)
    if isinstance(down, torch.Tensor) and isinstance(first, torch.Tensor) and down.ndim == 3 and first.ndim == 3:
        return first, down
    return None


def classify_model(model: nn.Module) -> Census:
    """Classify every weight-bearing module of ``model`` (see the module docstring)."""
    modules = dict(model.named_modules())

    def ancestors(path: str) -> list[str]:
        """Lower-cased class names of every enclosing module, nearest first."""
        parts = path.split(".")
        return [type(modules[".".join(parts[:i])]).__name__.lower() for i in range(len(parts) - 1, 0, -1)]

    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    emb_in = get_in() if callable(get_in) else None
    emb_out = get_out() if callable(get_out) else None

    experts: dict[str, int] = {}  # experts module path -> num_experts
    for n, m in modules.items():
        tensors = _expert_tensors(m)
        if tensors is not None:
            experts[n] = int(tensors[1].shape[0])
        elif isinstance(m, nn.ModuleList) and len(m) > 1 and n.rsplit(".", 1)[-1] in ("experts", "expert"):
            experts[n] = len(m)
    moe_blocks = {n.rsplit(".", 1)[0]: ne for n, ne in experts.items() if "." in n}

    census = Census(model_params=sum(t.numel() for t in model.parameters()))
    for n, m in modules.items():
        if emb_in is not None and m is emb_in:
            weight = getattr(m, "weight")
            census.entries.append(Entry(n, "embedding", "embedding", weight.numel(), int(weight.shape[-1])))
            continue
        tensors = _expert_tensors(m)
        if tensors is not None:
            first, down = tensors
            in_dim = min(int(first.shape[-1]), int(down.shape[-1]))
            census.entries.append(Entry(n, "routed_experts", "fused_experts", first.numel() + down.numel(), in_dim))
            continue
        if not isinstance(m, nn.Linear):
            continue
        anc = ancestors(n)
        parent = n.rsplit(".", 1)[0] if "." in n else ""
        if emb_out is not None and m is emb_out:
            cls = "lm_head"
        elif any(n.startswith(e + ".") for e in experts):
            cls = "routed_experts"
        elif parent in moe_blocks and m.out_features == moe_blocks[parent]:
            cls = "router"
        elif m.out_features < SMALL_OUT:
            cls = "small_proj"
        elif any(seg.startswith("shared_expert") for seg in n.split(".")):
            cls = "shared_experts"
        elif any(h in c for c in anc for h in LINEAR_ATTN_HINTS) or ".linear_attn." in f".{n}.":
            cls = "linear_attn"
        elif any("attention" in c for c in anc):
            cls = "attention"
        elif any(c.endswith("mlp") for c in anc):
            cls = "dense_mlp"
        else:
            cls = "other"
        census.entries.append(Entry(n, cls, "linear", m.weight.numel(), m.in_features))

    # Every parameter not claimed above is still accounted for: matrices as router/other entries (e.g. a
    # TopKRouter that is not an nn.Linear), 1-D vectors (norms, A_log, dt_bias, biases) as one aggregate.
    covered = set()
    for e in census.entries:
        covered.update(id(t) for t in modules[e.module].parameters())
    vec_params = 0
    for n, m in modules.items():
        for pn, t in m.named_parameters(recurse=False):
            if id(t) in covered:
                continue
            covered.add(id(t))
            if t.ndim < 2:
                vec_params += t.numel()
                continue
            parent = n.rsplit(".", 1)[0] if "." in n else ""
            is_router = n in moe_blocks or (parent in moe_blocks and t.shape[0] == moe_blocks[parent])
            cls = "router" if is_router and t.ndim == 2 and t.shape[0] in moe_blocks.values() else "other"
            kind = "conv" if "conv" in type(m).__name__.lower() else "param"
            census.entries.append(Entry(f"{n}.{pn}" if n else pn, cls, kind, t.numel(), int(t.shape[-1])))
    if vec_params:
        census.entries.append(Entry("<1-D vectors>", "vectors", "vectors", vec_params, 1))
    return census


def apply_policy(census: Census, policy: dict[str, str]) -> Census:
    """Choose each entry's format from ``policy``, falling back where the input dim does not fit a block."""
    for e in census.entries:
        want = policy.get(e.cls, "none")
        fmt = want
        while fmt != "none" and e.in_dim % FORMATS[fmt][0] != 0:
            fmt = FALLBACK[fmt]
        e.fmt = fmt
        if fmt != want:
            e.note = f"{want} wanted, in_dim {e.in_dim} not a multiple of {FORMATS[want][0]} -> {fmt}"
        if e.cls == "other" and want == "none" and e.kind != "conv":
            e.note = e.note or "unclassified weight left unquantized"
    return census


def _numerics(fmt: str) -> dict:
    block, bits = FORMATS[fmt]
    return {"num_bits": fmt, "block_sizes": {-1: block}, "backend": "ggml", "effective_bits": bits}


def build_quant_cfg(census: Census, algorithm: str | dict | None = "max") -> dict:
    """ModelOpt QuantizeConfig dict: everything disabled, then exact per-module weight quantizers enabled."""
    quant_cfg: list[dict] = [{"quantizer_name": "*", "enable": False}]
    quant_cfg.extend(
        {
            "quantizer_name": (
                f"{e.module}.*weight_quantizer*" if e.kind == "fused_experts" else f"{e.module}.weight_quantizer"
            ),
            "cfg": _numerics(e.fmt),
        }
        for e in census.entries
        if e.fmt != "none" and e.cls not in EXPORT_ONLY
    )
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


def run_census(model: nn.Module, policy: dict[str, str] | None = None) -> Census:
    """Classify and apply ``policy`` (default: Bolt's map)."""
    return apply_policy(classify_model(model), {**DEFAULT_POLICY, **(policy or {})})


def main() -> None:
    """Print the census of a model dir (meta-device load: no weights read) and optionally write it as JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", default="", help="comma list class=format overriding DEFAULT_POLICY")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    # Multimodal checkpoints (Qwen3_5*ForConditionalGeneration) carry a composite config; QAD is text-only
    # (vision tower and MTP head are not trained), so build the causal LM from the text config.
    config = AutoConfig.from_pretrained(a.model).get_text_config()
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    policy = dict(kv.split("=", 1) for kv in a.policy.split(",") if kv)
    census = run_census(model, policy)
    print(census.table())
    if a.json:
        a.json.write_text(json.dumps(census.to_json(), indent=1))


if __name__ == "__main__":
    main()
