#!/usr/bin/env python3
"""bm-889 sft2 gate (c) fix: move every source row with a 200-gram eval hit from eval to train, whole rows, so no turn is
lost and no sibling of a moved turn stays in eval. Eval is not refilled: a new pick could collide again. CPU.
Writes a new data dir (never overwrites). Usage: sft2_resplit.py RENDER DATA_IN DATA_OUT"""
import collections, glob, json, re, shutil, sys
from pathlib import Path
import numpy as np

src = open("sft2_overlap.py").read().split("render = Path")[0]
exec(src)
render, din, dout = sys.argv[1:4]
Path(dout).mkdir(exist_ok=False)
for f in glob.glob(f"{din}/*"):
    shutil.copy(f, dout)
train = np.unique(np.concatenate([grams(x) for f in sorted(glob.glob(f"{render}/*.train.roles.jsonl")) for x, _ in rows(f)]))
for f in sorted(glob.glob(f"{render}/*.eval.roles.jsonl")):
    fam = f.split("/")[-1].split(".")[0]
    known = [json.loads(l)["id"] for l in open(f"{din}/{fam}.eval.conv.jsonl")]
    bad = set()
    for line in open(f):
        if not line.strip():
            continue
        w = json.loads(line)
        x, r = np.asarray(w["ids"], np.int64), np.asarray(w["role"], np.int8)
        hit = np.isin(grams(x), train) & (r[N - 1 :] > 0)
        if not hit.any():
            continue
        members = [c for c in known if c in w["id"]]
        members.sort(key=w["id"].index)
        bos = np.cumsum(x == 2)
        for i in np.nonzero(hit)[0]:
            bad.add(members[max(int(bos[i + N - 1]) - 1, 0)])
    if not bad:
        continue
    keys = {c.rsplit("/t", 1)[0] for c in bad}
    lines = open(f"{din}/{fam}.eval.conv.jsonl").read().splitlines()
    stay = [l for l in lines if json.loads(l)["id"].rsplit("/t", 1)[0] not in keys]
    move = [l for l in lines if json.loads(l)["id"].rsplit("/t", 1)[0] in keys]
    Path(f"{dout}/{fam}.eval.conv.jsonl").write_text("\n".join(stay) + "\n")
    with open(f"{dout}/{fam}.train.conv.jsonl", "a") as ft:
        ft.write("\n".join(move) + "\n")
    print("MOVED", fam, "hit examples", len(bad), "rows", len(keys), "turns moved", len(move), "eval left", len(stay), flush=True)
print("RESPLIT_OK")
