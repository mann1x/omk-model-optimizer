#!/usr/bin/env python3
# find_overlap_rows.py RENDER EVAL_CONV -- which eval conversations of a packed eval render hold the scored positions that end
# a 200-gram seen in train (same hash as sft2_overlap.py). Prints conversation ids (from the window's '+'-joined names).
import glob, json, sys
import numpy as np
sys.argv, R = sys.argv, sys.argv[1]
N = 200; P = np.uint64(1099511628211)
def grams(a):
    if len(a) < N: return np.zeros(0, np.uint64)
    a = a.astype(np.uint64) + np.uint64(1); h = np.zeros(len(a) - N + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N): h = h * P + a[j:j + len(h)]
    return h
train = np.unique(np.concatenate([grams(np.asarray(json.loads(l)["ids"], np.int64)) for f in sorted(glob.glob(f"{R}/*.train.roles.jsonl")) for l in open(f) if l.strip()]))
for l in open(f"{R}/kimi31b.eval.roles.jsonl"):
    d = json.loads(l); x = np.asarray(d["ids"], np.int64); r = np.asarray(d["role"], np.int8)
    hit = np.isin(grams(x), train) & (r[N - 1:] > 0)
    if hit.any():
        bos = [i for i, t in enumerate(x) if t == 2]; names = d["id"].split("+")
        for p in np.nonzero(hit)[0] + N - 1:
            k = max(i for i, b in enumerate(bos) if b <= p); print(names[k], int(p))
