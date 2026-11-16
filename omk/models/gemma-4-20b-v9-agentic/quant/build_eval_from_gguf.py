"""bm-893: an eval reference set for a model that has no QAD cache — its own F16/BF16 GGUF's top-K via llama_topk.

Takes the rows (tokens x, role masks, families) of an existing eval.pt (the 26B QAD cache's 44 held-out role rows;
the 20B v9-agentic shares the Gemma tokenizer) and replaces idx/val with the given GGUF's top-K log-probs at every
scored position (role[1:] > 0, anchor as teacher), so step0_gguf_student_eval.py scores a quant of THIS model
against its own unquantised GGUF: the KL then measures the quantisation alone (same llama.cpp kernels on both sides).

Usage: build_eval_from_gguf.py --gguf F16.gguf --rows eval.pt --out eval_20b.pt --bin llama_topk --omk OMK [--topk 64]
Sentinel: EVAL_FROM_GGUF OK <scored> | FAIL. Never overwrites --out.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--rows", required=True, help="eval.pt whose x / role / family / families are reused")
ap.add_argument("--out", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--omk", required=True)
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--n-vocab", type=int, default=262144)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
if Path(a.out).exists():
    raise SystemExit(f"EVAL_FROM_GGUF FAIL {a.out} exists (never overwritten)")
sys.path.insert(0, a.omk)
from omk_qad.gguf_teacher import GGUFTeacher  # noqa: E402

ev = torch.load(a.rows, weights_only=False)
x_all, role_all = ev["x"], ev["role"]
n_rows, seq = x_all.shape
idx = torch.zeros(n_rows, seq - 1, a.topk, dtype=torch.int32)
val = torch.full((n_rows, seq - 1, a.topk), -1e4, dtype=torch.float16)
g = GGUFTeacher(a.gguf, seq + 8, a.topk, a.n_vocab, a.bin, ngl=999, device=a.device)
t0, scored, nll_sum = time.time(), 0, 0.0
for r in range(n_rows):
    x = x_all[r : r + 1].long()
    want = (role_all[r].long()[1:] > 0)[None]
    if not want.any():
        continue
    gi, gv, nll = g(x, want)
    gi = torch.as_tensor(np.asarray(gi[0]))
    gv = torch.as_tensor(np.asarray(gv[0].float() if torch.is_tensor(gv) else gv[0], np.float32))
    w = want[0]
    idx[r][w] = gi[w].to(torch.int32)
    val[r][w] = gv[w].to(torch.float16)
    scored += int(w.sum())
    if nll is not None:
        nn = np.asarray(nll[0] if hasattr(nll, "__len__") else nll, np.float64)
        nll_sum += float(nn[w.numpy()].sum()) if nn.ndim else 0.0
    print(f"row {r + 1}/{n_rows} scored {scored} {time.time() - t0:.0f}s", flush=True)
g.close()
sha = hashlib.sha256(Path(a.gguf).read_bytes()[: 1 << 24]).hexdigest()[:16]  # header fingerprint, not the full 38 GB
out = {"x": x_all, "idx": idx, "val": val, "role": role_all, "family": ev["family"], "families": ev["families"],
       "identity": f"gguf-top{a.topk}-{sha}", "source_gguf": a.gguf, "rows_from": a.rows,
       "teacher_ppl": float(np.exp(nll_sum / scored)) if nll_sum else None}
torch.save(out, a.out)
print(f"EVAL_FROM_GGUF OK {scored} rows {n_rows} topk {a.topk} out {a.out}", flush=True)
