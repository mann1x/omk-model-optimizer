#!/usr/bin/env python3
"""bm-889 sft2: turn-by-turn regeneration JOBS from the source rows (CPU only, no model calls).

A job is one assistant turn to regenerate with Gemma-4 itself: the recorded history before it (system, tools,
user, the ORIGINAL assistant calls/answers WITHOUT their foreign reasoning, the recorded tool results), the
recorded turn (what the validator compares a regenerated tool call against), its type and, for a final turn,
the reference answer where the source has one. Responses of the sources are never trained on.

Sources:
  hermes     the an-finetune v27 pool (interstellarninja/hermes_reasoning_tool_use, render-clean), normalized by
             an-finetune's replay_normalize.convert_hermes; its Gemma-native ``tool_responses`` (results embedded
             on the calling assistant turn) are split into OpenAI ``role:"tool"`` messages, the schema the server
             and omk_qad.rollouts take.
  <fetched>  sft2_fetch.py rows (Nemotron, already OpenAI schema with ``reasoning_content``); an empty system
             message is dropped; ``--drop-system`` sources lose their system turn (OpenCode's 9.9k-char agent
             boilerplate), keeping their tools.

Turn types: ``call`` (the recorded turn calls tools), ``nocall`` (a hermes ``relevance`` row whose recorded turn
answers without a call although tools are offered), ``final`` (the last assistant turn, no call), ``text`` (an
intermediate answer without a call). Rows with more than ``--max-turns`` assistant turns get a seeded sample of
that many, always including the last.

Out: OUT/jobs.jsonl, OUT/jobs.stats.json; ``--render-check N`` renders N jobs' prompts with the 26B tokenizer
(the served template) and counts failures.

Usage: sft2_jobs.py OUT --hermes POOL --rows DIR [--per-source N] [--max-turns 4] [--render-check 200]
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

AN = Path("/srv/ml/an-finetune/docker")
sys.path.insert(0, str(AN))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sft_build"))
from build_mix import norm_messages, tool_defs  # noqa: E402  (sft1's Nemotron normalizers: AI-SDK parts, MCP tools)
from replay_normalize import convert_hermes  # noqa: E402  (an-finetune's validated hermes converter)
from sft2_validate import check_calls  # noqa: E402  (the same schema check the regenerated calls must pass)

TOKENIZER = "/mnt/sdc/ml/models/gemma-4-26B-A4B-it"
DROP_SYSTEM = {"opencode"}
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def split_tool_responses(msgs: list[dict]) -> list[dict]:
    """Gemma-native ``tool_responses`` on an assistant turn -> OpenAI ``role:"tool"`` messages after it."""
    out = []
    for m in msgs:
        m = dict(m)
        resp = m.pop("tool_responses", None)
        out.append(m)
        if not resp:
            continue
        calls = m.get("tool_calls") or []
        for j, r in enumerate(resp):
            cid = calls[j]["id"] if j < len(calls) and calls[j].get("id") else f"call_r{j}"
            body = r.get("response") if isinstance(r, dict) else r
            out.append({"role": "tool", "tool_call_id": cid, "name": (r or {}).get("name"),
                        "content": body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)})
    return out


def fetched_messages(row: dict, drop_system: bool) -> tuple[list[dict], list | None]:
    """A fetched Nemotron row as OpenAI messages + functions (OpenCode: AI-SDK tool-result parts, MCP tools)."""
    msgs = []
    for m in row["messages"]:
        m = dict(m)
        c = m.get("content")
        if m.get("role") == "tool" and not m.get("tool_call_id") and isinstance(c, list):
            ids = [p.get("toolCallId") for p in c if isinstance(p, dict) and p.get("toolCallId")]
            if ids:
                m["tool_call_id"] = ids[0]
        if m.get("role") == "system" and (drop_system or not c):
            continue
        msgs.append(m)
    tools = [t for t in (tool_defs(row) or []) if t["function"]["name"] != "invalid"] or None
    return norm_messages(msgs), tools


def without_reasoning(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "reasoning_content"} if m["role"] == "assistant" else m


def reference(source: str, row: dict, recorded: dict) -> dict | None:
    """The answer a regenerated FINAL turn is checked against (None: termination + no loop only)."""
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if md.get("final_answer_entity"):
        return {"kind": "entity", "value": md["final_answer_entity"]}
    text = recorded.get("content") or ""
    b = BOXED.findall(text)
    if source == "science" and b:
        return {"kind": "mcq", "value": b[-1].strip()}
    if source == "math" and b:
        return {"kind": "boxed", "value": b[-1].strip()}
    return None


def turn_type(row_meta: dict, msgs: list[dict], k: int, last: int, tools) -> str:
    m = msgs[k]
    if m.get("tool_calls"):
        return "call"
    if tools and row_meta.get("scenario_category") == "relevance":
        return "nocall"
    return "final" if k == last else "text"


def jobs_of(source: str, ref: dict, msgs: list[dict], tools, meta: dict, row: dict, max_turns: int, rng) -> list[dict]:
    asst = [k for k, m in enumerate(msgs) if m["role"] == "assistant"]
    if not asst or not any(m["role"] == "user" for m in msgs[: asst[0]]):
        return []
    pick = asst if len(asst) <= max_turns else sorted(rng.sample(asst[:-1], max_turns - 1) + [asst[-1]])
    out = []
    for k in pick:
        t = turn_type(meta, msgs, k, asst[-1], tools)
        out.append({"job": f"{source}/{ref['id']}/t{k}", "source": source, "ref": ref, "turn": k,
                    "type": t, "context": [without_reasoning(m) for m in msgs[:k]], "tools": tools,
                    "recorded": msgs[k], "reference": reference(source, row, msgs[k]) if t == "final" else None,
                    "scenario": meta.get("scenario_category")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hermes", required=True, help="an-finetune hermes pool (render-clean jsonl)")
    ap.add_argument("--rows", required=True, help="sft2_fetch.py rows dir")
    ap.add_argument("--per-source", type=int, default=3000, help="rows per source (hermes: seeded sample)")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--render-check", type=int, default=0)
    ap.add_argument("--seed", type=int, default=889)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    stats: dict = {"rows": Counter(), "dropped": Counter(), "jobs": Counter(), "by_type": {}, "with_reference": Counter()}
    jobs = []

    lines = open(a.hermes).read().splitlines()
    for i in sorted(random.Random(f"{a.seed}/hermes").sample(range(len(lines)), min(a.per_source, len(lines)))):
        row = json.loads(lines[i])
        stats["rows"]["hermes"] += 1
        msgs, tools = convert_hermes(row)
        if not msgs:
            stats["dropped"]["hermes:convert"] += 1
            continue
        jobs += jobs_of("hermes", {"file": Path(a.hermes).name, "line": i, "id": f"L{i}"},
                        split_tool_responses(msgs), tools, row, row, a.max_turns, rng)

    for f in sorted(Path(a.rows).glob("*.jsonl")):
        src = f.stem
        for n, line in enumerate(open(f)):
            if n >= a.per_source:
                break
            rec = json.loads(line)
            row = rec["row"]
            stats["rows"][src] += 1
            if not isinstance(row.get("messages"), list):
                stats["dropped"][f"{src}:no_messages"] += 1
                continue
            try:
                msgs, tools = fetched_messages(row, src in DROP_SYSTEM)
            except ValueError as e:  # an unhandled content part: counted, never silent
                stats["dropped"][f"{src}:{str(e)[:40]}"] += 1
                continue
            ref = {"repo": rec["repo"], "rev": rec["rev"], "path": rec["path"], "offset": rec["offset"],
                   "sha256": rec["sha256"], "id": f"{rec['path']}@{rec['offset']}"}
            jobs += jobs_of(src, ref, msgs, tools, row.get("metadata") or {}, row, a.max_turns, rng)

    # A recorded call that is itself invalid (an undeclared tool, a mistyped or missing argument: 37/2,420 in the
    # 2026-10-29 smoke) cannot be regenerated as an equivalent VALID call: such a job is dropped, counted by reason.
    # One job per unique (context, tools): the hermes pool repeats prompts (9,178 first-user scenarios in 50,830
    # rows; in the 2026-10-29 smoke one first turn was shared by 20 rows, each with a different recorded answer).
    # Gemma regenerates its own answer, so a repeated context is only extra generation and extra weight.
    keep, seen = [], set()
    for j in jobs:
        bad = check_calls(j, j["recorded"].get("tool_calls") or []) if j["type"] == "call" else []
        ctx = hashlib.sha256(json.dumps([j["context"], j["tools"]], sort_keys=True, ensure_ascii=False).encode())
        if bad:
            stats["dropped"][f"{j['source']}:recorded_{bad[0].split(' ')[0]}"] += 1
        elif ctx.hexdigest() in seen:
            stats["dropped"][f"{j['source']}:duplicate_context"] += 1
        else:
            seen.add(ctx.hexdigest())
            keep.append(j)
    jobs = keep
    for j in jobs:
        stats["jobs"][j["source"]] += 1
        stats["by_type"].setdefault(j["source"], Counter())[j["type"]] += 1
        stats["with_reference"][j["source"]] += j["reference"] is not None
    with open(out / "jobs.jsonl", "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")

    if a.render_check:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        bad = Counter()
        sample = random.Random(f"{a.seed}/render").sample(jobs, min(a.render_check, len(jobs)))
        for j in sample:
            try:
                ctx = json.loads(json.dumps(j["context"]))
                for m in ctx:  # the template renders dict arguments (omk_qad.rollouts._norm)
                    for c in m.get("tool_calls") or []:
                        fn = c.get("function", c)
                        if isinstance(fn.get("arguments"), str):
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                tok.apply_chat_template(ctx, tools=j["tools"], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=True)
            except Exception as e:  # noqa: BLE001  (counted per source and kind, never silent)
                bad[f"{j['source']}:{type(e).__name__}"] += 1
        stats["render_check"] = {"n": len(sample), "failed": dict(bad)}
    stats = {k: (dict(v) if isinstance(v, Counter) else {s: dict(c) for s, c in v.items()} if k == "by_type" else v)
             for k, v in stats.items()}
    (out / "jobs.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("JOBS_OK", len(jobs))


if __name__ == "__main__":
    main()
