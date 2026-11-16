#!/usr/bin/env python3
"""Tensor-by-tensor comparison of two GGUFs of the same layout: type, bytes equal, and for differing tensors the
relative RMS difference after dequantisation (libggml to_float via gguf's reader is not available, so differing
tensors are reported by class and by how many bytes differ). bm-892: omk export vs llama-quantize at layout E.

Usage: gguf_tensor_diff.py A.gguf B.gguf
"""

import re
import sys
from collections import Counter, defaultdict

import numpy as np
from gguf import GGUFReader


def cls(name: str) -> str:
    """Tensor class for the summary."""
    if "attn_" in name and name.endswith(".weight") and "norm" not in name:
        return "attention"
    if "ffn_gate_up_exps" in name or "ffn_down_exps" in name:
        return "routed_experts"
    if re.search(r"ffn_(gate|up|down)\.weight$", name):
        return "dense_mlp"
    if "norm" in name:
        return "norms"
    if "token_embd" in name or "output.weight" in name:
        return "embd_or_head"
    return "other"


def main() -> None:
    """CLI."""
    a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    if only_a or only_b:
        print("only in A:", only_a[:10], "only in B:", only_b[:10])
    same, type_diff, byte_diff = Counter(), Counter(), Counter()
    frac = defaultdict(list)
    examples = defaultdict(list)
    for n in sorted(set(ta) & set(tb)):
        x, y = ta[n], tb[n]
        c = cls(n)
        if x.tensor_type != y.tensor_type:
            type_diff[c] += 1
            examples[("type", c)].append(f"{n} {x.tensor_type.name} vs {y.tensor_type.name}")
            continue
        xa = np.asarray(x.data).reshape(-1).view(np.uint8)
        ya = np.asarray(y.data).reshape(-1).view(np.uint8)
        if xa.size != ya.size:
            byte_diff[c] += 1
            examples[("size", c)].append(n)
            continue
        if np.array_equal(xa, ya):
            same[c] += 1
        else:
            byte_diff[c] += 1
            frac[(c, x.tensor_type.name)].append(float(np.count_nonzero(xa != ya)) / xa.size)
            if len(examples[("bytes", c)]) < 3:
                examples[("bytes", c)].append(f"{n} {x.tensor_type.name}")
    print("class            identical  type-differs  bytes-differ")
    for c in sorted(set(same) | set(type_diff) | set(byte_diff)):
        print(f"{c:16s} {same[c]:9d}  {type_diff[c]:12d}  {byte_diff[c]:12d}")
    for (c, t), fs in sorted(frac.items()):
        print(f"  bytes differing in {c}/{t}: n={len(fs)} mean frac {np.mean(fs):.3f} min {min(fs):.3f} max {max(fs):.3f}")
    for k, v in sorted(examples.items()):
        print(" ", k, v[:3])


if __name__ == "__main__":
    main()
