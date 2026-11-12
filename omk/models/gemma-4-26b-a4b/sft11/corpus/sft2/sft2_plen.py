#!/usr/bin/env python3
"""bm-889 sft2: prompt length of every job (served template, 26B tokenizer) and the window filter (CPU only).

A training row is the job's context plus the regenerated turn, trained in a --seq window (16384, as sft1). A job
whose rendered prompt leaves less than --min-gen tokens of that window cannot become a training row, so it is not
generated: dropped, counted per source. A job whose context ENDS with an assistant turn (hermes multistep rows with
two assistant turns back to back, OpenCode continuation turns: 239 in jobs_v1) asks for a turn after a turn: the
trainer's renderer refuses it (rollouts NotPrefixError, 15/548 hermes in the stub run), so it is dropped too.
Kept jobs carry ``prompt_tokens``; sft2_generate.py caps each request at
window - prompt_tokens (a turn that does not fit the window is not trainable either; such cap hits are counted).

Usage: sft2_plen.py JOBS OUT_JOBS [--window 16384] [--min-gen 4096] [--procs 16]
"""
import argparse
import json
from collections import Counter
from multiprocessing import Pool

TOKENIZER = "/mnt/sdc/ml/models/gemma-4-26B-A4B-it"
_tok = None


def plen(line: str) -> tuple[str, int]:
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    j = json.loads(line)
    ctx = json.loads(json.dumps(j["context"]))
    for m in ctx:  # the template renders dict arguments (omk_qad.rollouts._norm)
        for c in m.get("tool_calls") or []:
            fn = c.get("function", c)
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"] or "{}")
    text = _tok.apply_chat_template(ctx, tools=j["tools"], add_generation_prompt=True, tokenize=False,
                                    enable_thinking=True)
    return line, len(_tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--min-gen", type=int, default=4096)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    kept, dropped, after_assistant = Counter(), Counter(), Counter()
    with open(a.jobs) as fi, open(a.out, "w") as fo, Pool(a.procs) as pool:
        for line, n in pool.imap(plen, fi, chunksize=64):
            j = json.loads(line)
            if j["context"][-1]["role"] == "assistant":
                after_assistant[j["source"]] += 1
                continue
            if n > a.window - a.min_gen:
                dropped[j["source"]] += 1
                continue
            kept[j["source"]] += 1
            fo.write(json.dumps(j | {"prompt_tokens": n}, ensure_ascii=False) + "\n")
    stats = {"window": a.window, "min_gen": a.min_gen, "kept": dict(kept), "dropped_long_prompt": dict(dropped),
             "dropped_after_assistant": dict(after_assistant)}
    print(json.dumps(stats, indent=1))
    print("PLEN_OK", sum(kept.values()))


if __name__ == "__main__":
    main()
