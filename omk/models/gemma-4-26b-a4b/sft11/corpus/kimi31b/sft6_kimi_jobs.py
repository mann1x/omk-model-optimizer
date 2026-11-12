#!/usr/bin/env python3
"""bm-889 sft6 C8: kimi prompts -> turn-by-turn regeneration jobs for the Gemma-4 31B teacher (CPU only).

Owner 2026-11-04 12:00 CET: "the kimi traces should be rerendered with gemma 31b and only the short ones, we need to
get a few and limited to 8k or 12k max"; 12:12 CET: "start the C8 kimi regeneration with the 31B".

Rows: lambda hermes-agent-reasoning-traces (config kimi), the rows sft4's hermes validator passed
(sft4_build/validated/lambda_kimi.pass.jsonl) with no flags and a full render of <= --max-row-tokens (8192), minus
sft4's eval-overlap drop list; a seeded sample of --rows, stratified by category in proportion to what is left.
Jobs are built exactly as sft2_jobs.py builds hermes jobs (imported, not copied): an-finetune convert_hermes ->
split_tool_responses -> jobs_of (history assistant turns lose kimi's reasoning; kimi's text is never a training
target, only masked history), recorded calls that are themselves invalid are dropped, one job per unique context.

Out: OUT/jobs.jsonl, OUT/jobs.stats.json, OUT/rows.json (the selected row ids, for provenance).
Usage: sft6_kimi_jobs.py OUT [--rows 300] [--max-row-tokens 8192] [--max-turns 6] [--seed 889]
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

G = Path("/mnt/sdc/omk-model-optimizer/g4_recovery")
sys.path.insert(0, str(G / "sft2_build"))
import sft2_jobs as J  # noqa: E402  (convert_hermes, split_tool_responses, jobs_of, check_calls)

SRC = G / "sft4_build/src/lambda_kimi.parquet"
VALIDATED = G / "sft4_build/validated/lambda_kimi.pass.jsonl"
OVERLAP = G / "sft4_build/kimi_pre2r_eval_overlap_drop.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--max-row-tokens", type=int, default=8192)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=889)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a job set
    rng = random.Random(a.seed)

    table = pq.read_table(SRC).to_pylist()
    drop = {i.split("/", 1)[1] for i in json.load(open(OVERLAP))["ids"]}
    cand = [v for v in map(json.loads, open(VALIDATED))
            if not v["flags"] and v["tokens"] <= a.max_row_tokens and v["id"] not in drop]
    by_cat: dict = {}
    for v in cand:
        by_cat.setdefault(table[v["index"]]["category"], []).append(v)
    # proportional quota per category (largest remainder), seeded sample inside each
    total = len(cand)
    raw = {c: a.rows * len(vs) / total for c, vs in by_cat.items()}
    quota = {c: int(x) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - quota[c], reverse=True)[: a.rows - sum(quota.values())]:
        quota[c] += 1
    picked = []
    for c in sorted(by_cat):
        picked += random.Random(f"{a.seed}/{c}").sample(by_cat[c], min(quota[c], len(by_cat[c])))

    stats: dict = {"candidates": total, "by_category_candidates": {c: len(v) for c, v in by_cat.items()},
                   "quota": quota, "rows": len(picked), "dropped": Counter(), "jobs": 0, "by_type": Counter()}
    jobs = []
    for v in sorted(picked, key=lambda v: v["index"]):
        row = table[v["index"]]
        msgs, tools = J.convert_hermes(row)
        if not msgs:
            stats["dropped"]["convert"] += 1
            continue
        ref = {"file": SRC.name, "index": v["index"], "id": row["id"], "category": row["category"],
               "row_tokens": v["tokens"]}
        jobs += J.jobs_of("kimi31b", ref, J.split_tool_responses(msgs), tools, {}, row, a.max_turns, rng)

    keep, seen = [], set()
    for j in jobs:
        bad = J.check_calls(j, j["recorded"].get("tool_calls") or []) if j["type"] == "call" else []
        ctx = hashlib.sha256(json.dumps([j["context"], j["tools"]], sort_keys=True, ensure_ascii=False).encode())
        if bad:
            stats["dropped"][f"recorded_{bad[0].split(' ')[0]}"] += 1
        elif ctx.hexdigest() in seen:
            stats["dropped"]["duplicate_context"] += 1
        else:
            seen.add(ctx.hexdigest())
            keep.append(j)
    for j in keep:
        stats["by_type"][j["type"]] += 1
    stats["jobs"] = len(keep)
    with open(out / "jobs.jsonl", "w") as fh:
        for j in keep:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    (out / "rows.json").write_text(json.dumps([{"index": v["index"], "id": v["id"], "tokens": v["tokens"]}
                                               for v in picked], indent=0))
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    (out / "jobs.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("KIMI_JOBS_OK", len(keep))


if __name__ == "__main__":
    main()
