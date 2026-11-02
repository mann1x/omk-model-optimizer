#!/usr/bin/env python3
"""bm-889 sft2: train/eval gate on a render dir: no scored token of an eval split may end a 200-gram seen in train (CPU).

Writes RENDER/OVERLAP_OK only when every family is at zero; sft2_train2.sh refuses a render without it.
Usage: sft2_overlap.py RENDER_DIR
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

N = 200
P = np.uint64(1099511628211)


def grams(a):
    if len(a) < N:
        return np.zeros(0, np.uint64)
    a = a.astype(np.uint64) + np.uint64(1)
    h = np.zeros(len(a) - N + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N):
            h = h * P + a[j : j + len(h)]
    return h


def rows(f):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            yield np.asarray(r["ids"], np.int64), np.asarray(r["role"], np.int8)


render = Path(sys.argv[1])
train = np.unique(np.concatenate([grams(x) for f in sorted(glob.glob(f"{render}/*.train.roles.jsonl")) for x, _ in rows(f)]))
total = 0
for f in sorted(glob.glob(f"{render}/*.eval.roles.jsonl")):
    hit = tot = 0
    for x, r in rows(f):
        sc = r[N - 1 :] > 0
        hit += int((np.isin(grams(x), train) & sc).sum())
        tot += int(sc.sum())
    total += hit
    print(f.split("/")[-1].removesuffix(".roles.jsonl"), f"n{N} {hit}/{tot}", flush=True)
if total:
    sys.exit(f"OVERLAP_FAIL {total} scored positions")
(render / "OVERLAP_OK").write_text("0\n")
print("OVERLAP_OK")
