#!/usr/bin/env python3
"""Build mix_v1 (bm-877): small code-heavy sources for the omk_qad Qwen3.5-4B smoke, from data already on bs2.

Each Nemotron source is a seeded random sample of whole rows (line offsets from a full scan, so long rows are not
over-picked), rendered with the student's own chat template (tools included) into {"text": ...}, capped at
--cap tokens per document, and filled until it holds >= budget * --margin tokens, so omk_qad's packer never
restarts (repeats) a source within one run. Eval rows are drawn first and are disjoint from the train rows.

Out: ROOT/data/mix_v1/<source>.train.jsonl, <source>.eval.jsonl, manifest.json (budgets, real token counts, row
indices, sha256 of every output, upstream repo sha from the stager's manifest).
"""
import argparse
import hashlib
import json
import mmap
import random
from pathlib import Path

NEM = "data/nemotron"
OC = "SFT-OpenCode-v1"
CP = "SFT-Competitive-Programming-v2/data"
# OpenCode rows open with the same ~9.9k-char opencode agent prompt (only the <env> path differs) and placeholder
# tool schemas: rendered whole, every document is mostly that boilerplate, so the system turn and tools are dropped.
DROP_SYSTEM = {"opencode"}
SOURCES = {  # name -> (files under data/nemotron, train token budget)
    "opencode": ([f"{OC}/{d}/data.jsonl" for d in ("general", "bash_only_tool", "bash_only_tool_skills",
                                                   "agent_skills", "agent_skills_question_tool", "question_tool")],
                 3_000_000),
    "swe_agentless": (["SFT-SWE-v2/data/agentless.jsonl"], 1_500_000),
    "cp_reasoning": ([f"{CP}/competitive_programming_{x}.jsonl.windows.jsonl"
                      for x in ("python_00", "python_01", "cpp_00", "cpp_01")], 1_500_000),
    "exercism_sql": ([f"{CP}/exercism.jsonl", f"{CP}/text_to_sql.jsonl"], 1_000_000),
}


def offsets(path: Path) -> list[int]:
    out, pos = [], 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        n = len(m)
        while pos < n:
            out.append(pos)
            j = m.find(b"\n", pos)
            pos = n if j < 0 else j + 1
    return out


def row_at(handles, fi, off):
    h = handles[fi]
    h.seek(off)
    return json.loads(h.readline())


def tool_defs(row) -> list | None:
    """OpenCode ships MCP-style {id, description, inputSchema.jsonSchema}; the template wants OpenAI functions."""
    tools = row.get("tools") or []
    out = []
    for t in tools:
        if "function" in t:
            out.append(t)
        elif "id" in t:
            out.append({"type": "function", "function": {
                "name": t["id"], "description": t.get("description", ""),
                "parameters": (t.get("inputSchema") or {}).get("jsonSchema") or {"type": "object"}}})
    return out or None


DEMO_USER = "Change the greeting to be more casual"


def norm_messages(msgs):
    """Tool-call arguments as dicts, reasoning carried once, and the fixed exercism demo dropped."""
    if any(m["role"] == "user" and m.get("content") == DEMO_USER for m in msgs):
        # exercism: every row repeats one system prompt + a demo user/assistant pair (~1k tokens of boilerplate)
        i = next(i for i, m in enumerate(msgs) if m["role"] == "user" and m.get("content") == DEMO_USER)
        msgs = msgs[i + 2:]
    out = []
    for m in msgs:
        m = dict(m)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    fn["arguments"] = {"raw": fn["arguments"]}
        c = m.get("content")
        if isinstance(c, list):  # OpenCode tool results: AI-SDK parts [{type: tool-result, output: {type, value}}]
            parts = []
            for it in c:
                if isinstance(it, dict) and it.get("type") == "tool-result":
                    v = (it.get("output") or {}).get("value")
                    parts.append(v if isinstance(v, str) else json.dumps(v))
                elif isinstance(it, dict) and it.get("type") == "text":
                    parts.append(it.get("text", ""))
                else:
                    raise ValueError(f"unhandled content part {str(it)[:80]}")
            c = m["content"] = "\n".join(parts)
        elif c is None:
            c = m["content"] = ""
        if m["role"] == "assistant" and isinstance(c, str) and "</think>" in c and m.get("reasoning_content"):
            m["content"] = c.split("</think>", 1)[1].lstrip("\n")  # reasoning is carried once, by reasoning_content
        out.append(m)
    return out


class PromptTooLong(ValueError):
    pass


def render(row, tok, cap: int, drop_system: bool = False):
    msgs = norm_messages(row["messages"])
    if drop_system:
        msgs = [m for m in msgs if m["role"] != "system"]
    text = tok.apply_chat_template(msgs, tools=None if drop_system else tool_defs(row), tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    first = text.find("<|im_start|>assistant")
    if first < 0 or len(tok(text[:first], add_special_tokens=False).input_ids) > cap // 2:
        # a document that is (almost) all prompt, e.g. a repository tree, carries no response to distil
        raise PromptTooLong(f"first assistant turn after {cap // 2} tokens")
    if len(ids) > cap:
        ids = ids[:cap]
        text = tok.decode(ids)
    return text, len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/sdc/omk-model-optimizer"))
    ap.add_argument("--tokenizer", default="/mnt/sdc/omk-model-optimizer/models/Qwen3.5-4B")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--margin", type=float, default=1.05)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--only", default=",".join(SOURCES))
    ap.add_argument("--preview", type=int, default=0, help="print N rendered docs per source and stop")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out_dir = a.root / "data" / "mix_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    up = json.loads((a.root / NEM / "manifest.json").read_text())
    man_p = out_dir / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"sources": {}}
    man.update(tokenizer=a.tokenizer, cap=a.cap, margin=a.margin, seed=a.seed)
    for name in a.only.split(","):
        files, budget = SOURCES[name]
        rng = random.Random(f"{a.seed}:{name}")
        pool = []
        for fi, f in enumerate(files):
            p = a.root / NEM / f
            pool += [(fi, o) for o in offsets(p)]
            print(f"{name}: {f} rows={len(pool)}", flush=True)
        rng.shuffle(pool)
        handles = [open(a.root / NEM / f, "rb") for f in files]
        if a.preview:
            for fi, o in pool[: a.preview]:
                try:
                    t, n = render(row_at(handles, fi, o), tok, a.cap, name in DROP_SYSTEM)
                except PromptTooLong as e:
                    print(f"--- {name} {Path(files[fi]).name}@{o} SKIP {e}")
                    continue
                marks = {k: t.count(k) for k in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant",
                                                  "<think>", "</think>", "<tools>", "<tool_call>", "<tool_response>")}
                print(f"--- {name} {Path(files[fi]).name}@{o} tokens={n} {marks}")
                print("    head:", repr(t[:160]))
            continue
        stats = {}
        it = iter(pool)
        for split, target in (("eval", None), ("train", int(budget * a.margin))):
            path = out_dir / f"{name}.{split}.jsonl"
            toks = docs = bad = long_prompt = 0
            used = []
            with open(path, "w") as o:
                for fi, off in it:
                    try:
                        t, n = render(row_at(handles, fi, off), tok, a.cap, name in DROP_SYSTEM)
                    except PromptTooLong:
                        long_prompt += 1
                        continue
                    except Exception as e:  # noqa: BLE001 - a row the template cannot render is skipped and counted
                        bad += 1
                        if bad <= 3:
                            print(f"   skip {files[fi]}@{off}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        continue
                    o.write(json.dumps({"text": t}) + "\n")
                    toks += n
                    docs += 1
                    used.append([fi, off])
                    if (target is None and docs >= a.n_eval) or (target is not None and toks >= target):
                        break
            if target is not None and toks < target:
                raise SystemExit(f"{name}: only {toks} tokens < {target}")
            stats[split] = {"docs": docs, "tokens": toks, "skipped": bad, "skipped_long_prompt": long_prompt,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": used}
            print(f"{name}.{split}: docs={docs} tokens={toks} skipped={bad} long_prompt={long_prompt}", flush=True)
        upstream = {s: up[s.split("/")[0]]["sha"] for s in {f.split("/")[0] for f in files} if s in up}
        man["sources"][name] = {"files": files, "budget": budget, "drop_system": name in DROP_SYSTEM,
                                "upstream_sha": upstream, **stats}
        man_p.write_text(json.dumps(man, indent=1))
        for h in handles:
            h.close()
    print("MIX_DONE", flush=True)


if __name__ == "__main__":
    main()
