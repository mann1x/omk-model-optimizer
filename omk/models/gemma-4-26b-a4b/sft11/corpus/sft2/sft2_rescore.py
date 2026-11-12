#!/usr/bin/env python3
"""bm-889 sft2: re-validate a generation dir's stored responses with the current validator (no model calls).

Each attempt's response (attempts.jsonl ``response`` for rejected ones, kept.jsonl for the passing one) is normalised
(sft2_generate.norm_message) and validated again; per job the FIRST passing attempt is kept, as the generator does.
Attempts whose response was not stored (before rejected responses were logged) are counted as unknown.
Out: OUT_DIR/attempts.jsonl (re-scored), OUT_DIR/kept.jsonl. Usage: sft2_rescore.py GEN_DIR JOBS OUT_DIR
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from sft2_generate import norm_message
from sft2_validate import validate

gen, jobs_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
kept = {}
for line in open(gen / "kept.jsonl"):
    k = json.loads(line)
    kept[k["job"]] = k["response"]
attempts = defaultdict(list)
for line in open(gen / "attempts.jsonl"):
    a = json.loads(line)
    attempts[a["job"]].append(a)
jobs = {}
for line in open(jobs_path):
    j = json.loads(line)
    if j["job"] in attempts:
        jobs[j["job"]] = j
n_unknown = 0
with open(out / "attempts.jsonl", "w") as fa, open(out / "kept.jsonl", "w") as fk:
    for jid, atts in attempts.items():
        job, done = jobs[jid], False
        for a in sorted(atts, key=lambda x: x["attempt"]):
            resp = a.get("response") or (kept.get(jid) if a["ok"] else None)
            if resp is None:
                n_unknown += 1
                fa.write(json.dumps(a | {"rescored": False}) + "\n")
                continue
            msg = norm_message(resp) | {"finish_reason": resp.get("finish_reason")}
            reasons = validate(job, msg)
            row = {k: v for k, v in a.items() if k != "response"} | {"ok": not reasons, "reasons": reasons,
                                                                       "rescored": True}
            fa.write(json.dumps(row | ({"response": msg} if reasons else {})) + "\n")
            if not reasons and not done:
                done = True
                fk.write(json.dumps({k: job[k] for k in ("job", "source", "type", "ref", "turn", "context", "tools",
                                                         "recorded")} | {"response": msg}) + "\n")
print("RESCORE_OK unknown", n_unknown)
