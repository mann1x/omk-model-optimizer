#!/usr/bin/env python3
# loopscan.py -- bm-889 uniform loop re-score (owner 2026-11-11 10:54: "re-score the previous sft and find out if there's at
#   least one that removed the looping"). Reads one run's `automaticum logs <id> --events -json` on stdin, rebuilds every
#   reasoning block (content_end of contentType reasoning; content_start text if a block never ended) and scores each block
#   the same way whatever the engine or SDK guard did:
#   units = lines and sentences of >= 40 chars, normalised (lower case, letters/digits only) and cut to an 80-char key, so a
#   long code line re-quoted with small edits stays one unit (the sft6 brace-count loop: line 90 written 70x in one block).
#   Per block: chars, units, maxrep (most repeats of one key), repshare (share of units that repeat an earlier key),
#   tailshare (repshare over the last quarter of the block's units).
# Prints one JSON line: run id + per-block list + run summary. Reads only; writes nothing but stdout.
import json
import re
import sys
from collections import Counter

run = sys.argv[1]
blocks, cur = [], []
cuts = 0
for line in sys.stdin:
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("k") == "runner" and e.get("note") == "guard_trip" and str(e.get("guard", "")).startswith("reasoning_loop"):
        cuts += 1
    if e.get("k") != "event":
        continue
    ev = e.get("event") or {}
    if ev.get("type") != "agent_event":
        continue
    a = (ev.get("payload") or {}).get("event") or {}
    if a.get("contentType") != "reasoning":
        continue
    t = a.get("type")
    if t == "content_start":
        cur.append(a.get("reasoning") or "")
    elif t == "content_end":
        blocks.append(a.get("reasoning") or "".join(cur))
        cur = []
if cur:
    blocks.append("".join(cur))

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
NORM = re.compile(r"[^a-z0-9]+")


def score(b):
    units = [u.strip() for u in SPLIT.split(b) if len(u.strip()) >= 40]
    keys = [NORM.sub("", u.lower())[:80] for u in units]
    keys = [k for k in keys if len(k) >= 20]
    n = len(keys)
    if not n:
        return {"chars": len(b), "units": 0, "maxrep": 0, "repshare": 0.0, "tailshare": 0.0, "top": ""}
    seen, rep, tail_rep, tail_n = Counter(), 0, 0, 0
    for i, k in enumerate(keys):
        r = seen[k] > 0
        rep += r
        if i >= 3 * n // 4:
            tail_n += 1
            tail_rep += r
        seen[k] += 1
    k, c = seen.most_common(1)[0]
    return {"chars": len(b), "units": n, "maxrep": c, "repshare": round(rep / n, 3),
            "tailshare": round(tail_rep / tail_n, 3) if tail_n else 0.0, "top": k[:60]}


S = [score(b) for b in blocks]
out = {"run": run, "blocks": len(S), "guard_cut_notes": cuts,
       "think_chars": sum(s["chars"] for s in S),
       "max_maxrep": max((s["maxrep"] for s in S), default=0),
       "blocks_maxrep_ge10": sum(s["maxrep"] >= 10 for s in S),
       "blocks_maxrep_ge20": sum(s["maxrep"] >= 20 for s in S),
       "chars_in_maxrep_ge10": sum(s["chars"] for s in S if s["maxrep"] >= 10),
       "per_block": S}
print(json.dumps(out))
