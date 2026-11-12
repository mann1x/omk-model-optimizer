#!/usr/bin/env python3
"""bm-889 sft2: kept regenerated turns -> conversations for omk_qad.rollouts, train/eval split per source (CPU).

Each kept job (sft2_generate.py kept.jsonl) is one conversation: the job's context with every history assistant
turn marked ``"score": false`` (written by another model, context only) followed by Gemma's regenerated turn (the
only scored turn). The eval split is by SOURCE ROW, not by job: all turns regenerated from one source row land in
the same split, seeded, until each source has at least ``--eval-min`` eval conversations (sft1's ``*_eval`` role).

Render afterwards with ``python -m omk_qad.rollouts --in <x>.conv.jsonl --history-reasoning strip --seq 16384
--pack`` (the generation context had no history reasoning, so strip is what the model saw).

Out: OUT/<source>.train.conv.jsonl, OUT/<source>.eval.conv.jsonl, OUT/assemble.stats.json; ASSEMBLE_OK.
Usage: sft2_assemble.py KEPT OUT [--eval-min 64] [--seed 889]
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TURN_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def row_id(job: dict) -> str:
    return job["job"].rsplit("/t", 1)[0]


def conversation(job: dict) -> dict:
    ctx = [m | {"score": False} if m["role"] == "assistant" else m for m in job["context"]]
    resp = {k: job["response"][k] for k in TURN_KEYS if job["response"].get(k)}
    resp["role"] = "assistant"
    return {"id": job["job"], "messages": [*ctx, resp], "tools": job["tools"],
            "meta": {"source": job["source"], "type": job["type"], "ref": job["ref"], "turn": job["turn"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kept")
    ap.add_argument("out")
    ap.add_argument("--eval-min", type=int, default=64)
    ap.add_argument("--seed", type=int, default=889)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)  # an assembled set is never overwritten
    by = defaultdict(lambda: defaultdict(list))  # source -> row id -> jobs
    for line in open(a.kept):
        j = json.loads(line)
        by[j["source"]][row_id(j)].append(j)
    stats = {"train": Counter(), "eval": Counter(), "eval_rows": Counter(), "by_type": {}}
    for src, rows in sorted(by.items()):
        order = sorted(rows, key=lambda r: hashlib.sha256(f"{a.seed}/{r}".encode()).hexdigest())
        ev, n = set(), 0
        for r in order:
            if n >= a.eval_min or len(ev) >= len(order) // 4:  # never more than a quarter of the rows
                break
            ev.add(r)
            n += len(rows[r])
        with open(out / f"{src}.train.conv.jsonl", "w") as ft, open(out / f"{src}.eval.conv.jsonl", "w") as fe:
            for r in order:
                for j in rows[r]:
                    split = "eval" if r in ev else "train"
                    (fe if split == "eval" else ft).write(json.dumps(conversation(j), ensure_ascii=False) + "\n")
                    stats[split][src] += 1
                    stats["by_type"].setdefault(src, Counter())[f"{split}:{j['type']}"] += 1
        stats["eval_rows"][src] = len(ev)
    stats = {k: {s: dict(c) for s, c in v.items()} if k == "by_type" else dict(v) for k, v in stats.items()}
    (out / "assemble.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("ASSEMBLE_OK", sum(stats["train"].values()), sum(stats["eval"].values()))


if __name__ == "__main__":
    main()
