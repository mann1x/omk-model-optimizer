"""Verify an imatrix (GGUF-format .dat as llama-imatrix writes it): KV chunk count / datasets, tensor coverage, finite
sums. Usage: python -I check_imatrix_gguf.py <imatrix.dat> [<reference imatrix.dat>]  (prints a one-line verdict).
"""
import sys

import numpy as np
from gguf import GGUFReader


def summary(path):
    r = GGUFReader(path)
    kv = {}
    for k, f in r.fields.items():
        if k.startswith("imatrix.") or k == "general.type":
            p = f.parts[f.data[0]]
            kv[k] = bytes(p).decode() if f.types[0].name == "STRING" else (int(p[0]) if p.dtype.kind in "iu" else p[0])
    names = [t.name for t in r.tensors]
    in_sum2 = [t for t in r.tensors if t.name.endswith(".in_sum2")]
    counts = [t for t in r.tensors if t.name.endswith(".counts")]
    bad = [t.name for t in in_sum2 if not np.isfinite(np.asarray(t.data, dtype=np.float32)).all()]
    zero_cols = sum(int((np.asarray(t.data, dtype=np.float32) == 0).sum()) for t in in_sum2)
    total_cols = sum(int(np.asarray(t.data).size) for t in in_sum2)
    experts = sorted({n.split(".")[1] for n in names if n.startswith("blk.") and "exps" in n})
    return kv, names, len(in_sum2), len(counts), bad, zero_cols, total_cols, experts


kv, names, n_in, n_cnt, bad, zc, tc, exps = summary(sys.argv[1])
print(f"KV: {kv}")
print(f"tensors {len(names)}: in_sum2 {n_in}, counts {n_cnt}; expert layers {len(exps)}; non-finite {bad}; "
      f"zero columns {zc}/{tc} ({100 * zc / max(tc, 1):.3f} %)")
exps_names = [n for n in names if "exps" in n and n.endswith(".in_sum2")]
print(f"routed-expert in_sum2 tensors: {len(exps_names)}, e.g. {exps_names[:2]}")
ok = n_in > 0 and n_in == n_cnt and not bad
if len(sys.argv) > 2:
    kv2, names2, n2, c2, bad2, _, _, _ = summary(sys.argv[2])
    same = sorted(names) == sorted(names2)
    print(f"reference {sys.argv[2]}: KV {kv2}; same tensor set {same}")
    ok = ok and same
print("IMATRIX_CHECK", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
