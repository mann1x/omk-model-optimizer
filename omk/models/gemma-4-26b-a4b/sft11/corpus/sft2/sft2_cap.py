#!/usr/bin/env python3
"""bm-889 sft2: the generation job set = a seeded per-(source, type) sample of the window-filtered jobs (CPU).

Caps are a JSON object {"source": N} or {"source/type": N} (a source/type cap wins over its source cap; a source
without a cap keeps all its jobs; 0 drops it). Within a source cap, jobs are taken in a seeded order with call /
nocall turns first (the target is the tool-calling degeneration), then the rest.

Usage: sft2_cap.py JOBS OUT --caps '{"hermes": 99999, "math": 800, "swe/final": 200}' [--seed 889]
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--caps", required=True)
    ap.add_argument("--seed", type=int, default=889)
    a = ap.parse_args()
    caps = json.loads(a.caps)
    by = defaultdict(list)
    for line in open(a.jobs):
        j = json.loads(line)
        by[j["source"]].append(j)
    out, stats = [], {}
    for src, js in sorted(by.items()):
        random.Random(f"{a.seed}/cap/{src}").shuffle(js)
        js.sort(key=lambda j: j["type"] not in ("call", "nocall"))  # stable: tool turns first, seeded within
        taken, per_type = [], Counter()
        for j in js:
            t_cap = caps.get(f"{src}/{j['type']}")
            if t_cap is not None and per_type[j["type"]] >= t_cap:
                continue
            if len(taken) >= caps.get(src, len(js)):
                break
            taken.append(j)
            per_type[j["type"]] += 1
        out += taken
        stats[src] = {"available": len(js), "taken": len(taken), "by_type": dict(per_type)}
    with open(a.out, "w") as fh:
        for j in out:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(json.dumps({"caps": caps, "sources": stats}, indent=1))
    print("CAP_OK", len(out))


if __name__ == "__main__":
    main()
