# SPDX-License-Identifier: Apache-2.0
"""General replay for bf16 recovery: on-policy chat answers of the student, in request-capture form.

The agentic pool teaches the loop fix; the replay keeps everything else (math, science, code, instruction
following, tool calls, multi-turn chat) from drifting. Its answers are the STUDENT's own, sampled on the serving
stack at the deployment sampler, so the teacher scores the trajectories the student actually produces (on-policy,
like the pool). Output is the Cerebriline SDK's request-capture layout, so ``omk_qad.rollouts --capture`` builds
the windows exactly as it does for the pool (cut responses, ``--verify-server``, roles: prompt 0, generated 1,
fenced code 2).

  gate-ngrams  word 13-grams of the acceptance benches (GPQA, MATH-500, HumanEval(+), LiveCodeBench) from the
               local HF datasets cache -> a JSON list of hashes (run where the cache is, ship the file)
  prompts      seeded per-category sample of Nemotron RL-blend rows (``responses_create_params``) -> chat bodies;
               a row is dropped when at least ``--gate-frac`` of its 13-grams are gate 13-grams (a copied problem
               shares ~80 %); a smaller share is shared boilerplate ("it is guaranteed that the sum of n over all
               test cases ...", digit grids) and is kept, counted as gate_partial_kept
  generate     POST each body to an OpenAI-compatible server -> ``<out>/<category>/<id>.json`` +
               ``.response.json``; resumable (a complete response is never re-asked), ``--parallel`` requests

    python -m omk_qad.replay gate-ngrams --out gate13.json
    python -m omk_qad.replay prompts --src 'RL-*/rlvr*.jsonl' --gate gate13.json --per-category 75 --out p.jsonl
    python -m omk_qad.replay generate --prompts p.jsonl --url http://host:port --model NAME --out CAPDIR
"""

import argparse
import glob
import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Nemotron RL-blend verifier names -> replay category. Left out on purpose: genrm (judging other answers),
# jailbreak / abstention / over-refusal (policy, not capability), lean, swe pivots (foreign agent histories),
# placeholder rows (``_hf_placeholder``: text only after fill_placeholders.py).
CATEGORIES = {
    "math_with_judge_simple_agent": "math",
    "mcqa_simple_agent": "science_mcqa",
    "code_gen_simple_agent": "code",
    "instruction_following_simple_agent": "instruction",
    "structured_outputs_simple_agent": "structured",
    "reasoning_gym_simple_agent": "reasoning",
    "single_step_tool_use_with_argument_comparison_agent": "tool_call",
    "multichallenge_simple_agent": "multiturn",
}
# The pool arm's sampler (automaticum g4-128e-bf16-c13-shipped: Google's defaults, no penalties).
SAMPLER = {"temperature": 1, "top_p": 0.95, "top_k": 64, "min_p": 0, "presence_penalty": 0,
           "repeat_penalty": 1, "repeat_last_n": 0}
N = 13
WORD = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int = N) -> set[str]:
    """Hashed lowercase word n-grams of a text."""
    w = WORD.findall(text.lower())
    return {hashlib.blake2b(" ".join(w[i : i + n]).encode(), digest_size=8).hexdigest()
            for i in range(len(w) - n + 1)}


# ===================================================================================================== gate
GATE_SETS = [  # (dataset, config, split, text fields): every split the acceptance benches draw from
    ("Idavidrein/gpqa", "gpqa_main", "train", ["Question", "Correct Answer"]),
    ("Idavidrein/gpqa", "gpqa_diamond", "train", ["Question", "Correct Answer"]),  # a subset of main: a check
    ("HuggingFaceH4/MATH-500", None, "test", ["problem"]),
    ("openai/openai_humaneval", None, "test", ["prompt"]),
    ("evalplus/humanevalplus", None, "test", ["prompt"]),
]
# LiveCodeBench: its raw release files (test.jsonl .. test6.jsonl = v1..v6) from the hub cache, every release
LCB_GLOB = "~/.cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/*/test*.jsonl"


def gate_ngrams(a) -> None:
    """13-grams of every gate bench text in the local datasets cache (offline; a missing set refuses)."""
    import os

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    grams, census = set(), {}
    for name, cfg, split, fields in GATE_SETS:
        ds = load_dataset(name, cfg, split=split)
        n0 = len(grams)
        for r in ds:
            for f in fields:
                grams |= ngrams(str(r.get(f) or ""))
        census[f"{name}:{cfg or '-'}"] = {"rows": len(ds), "new_ngrams": len(grams) - n0}
    lcb = {}  # question_id -> text: the releases overlap
    for f in sorted(glob.glob(os.path.expanduser(LCB_GLOB))):
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                lcb[r["question_id"]] = r["question_content"]
    if not lcb:
        raise SystemExit(f"no LiveCodeBench release files at {LCB_GLOB}")
    n0 = len(grams)
    for t in lcb.values():
        grams |= ngrams(t)
    census["livecodebench/code_generation_lite:v1-v6"] = {"rows": len(lcb), "new_ngrams": len(grams) - n0}
    Path(a.out).write_text(json.dumps({"n": N, "census": census, "ngrams": sorted(grams)}))
    print(json.dumps(census, indent=1))


# ===================================================================================================== prompts
def _text(content) -> str:
    """Responses-API content (string or list of typed parts) -> plain text."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def to_chat(p: dict) -> dict | None:
    """``responses_create_params`` -> a chat-completions body, or None when it is not a fresh user turn.

    Foreign reasoning items are dropped (strip); earlier tool calls / outputs stay as context.
    """
    msgs: list[dict] = []
    for it in p.get("input") or []:
        kind = it.get("type")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            call = {"id": it.get("call_id") or f"call_{len(msgs)}", "type": "function",
                    "function": {"name": it["name"], "arguments": it.get("arguments") or "{}"}}
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"), "content": _text(it.get("output"))})
        elif it.get("role") in ("system", "developer", "user", "assistant"):
            role = "system" if it["role"] == "developer" else it["role"]
            msgs.append({"role": role, "content": _text(it.get("content"))})
        else:
            return None
    if not msgs or msgs[-1]["role"] != "user":
        return None
    body: dict = {"messages": msgs}
    tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
             for t in p.get("tools") or [] if t.get("type") == "function"]
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    return body


def prompts(a) -> None:
    """Seeded per-category sample of the RL-blend rows, gate-decontaminated (see module doc)."""
    gate = set(json.loads(Path(a.gate).read_text())["ngrams"]) if a.gate else None
    if gate is None and not a.no_gate:
        raise SystemExit("--gate is required (or --no-gate to sample without decontamination)")
    files = sorted(f for g in a.src.split(",") for f in glob.glob(g))
    if not files:
        raise SystemExit(f"no files match {a.src}")
    pool: dict[str, list] = {}
    census = Counter()
    for f in files:
        with open(f) as fh:
            for i, ln in enumerate(fh):
                r = json.loads(ln)
                cat = CATEGORIES.get((r.get("agent_ref") or {}).get("name"))
                if cat is None:
                    continue
                if "responses_create_params" not in r:
                    census[f"{cat}:placeholder"] += 1
                    continue
                body = to_chat(r["responses_create_params"])
                if body is None:
                    census[f"{cat}:not_a_user_turn"] += 1
                    continue
                pool.setdefault(cat, []).append((f"{Path(f).parent.name}/{Path(f).stem}/{i}", body))
    rng = random.Random(a.seed)
    out = []
    for cat in sorted(pool):
        rows = pool[cat]
        rng.shuffle(rows)
        k = 0
        for rid, body in rows:
            if k == a.per_category:
                break
            text = "\n".join(m.get("content") or "" for m in body["messages"])
            if gate is not None:
                g = ngrams(text)
                hit = len(g & gate)
                if hit and hit >= a.gate_frac * len(g):
                    census[f"{cat}:gate_overlap"] += 1
                    continue
                census[f"{cat}:gate_partial_kept"] += int(hit > 0)
            out.append({"id": f"{cat}/{hashlib.sha1(rid.encode()).hexdigest()[:12]}", "category": cat,
                        "source": rid, "body": body})
            k += 1
        census[f"{cat}:kept"] = k
        census[f"{cat}:available"] = len(rows)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    meta = {"files": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}, "seed": a.seed,
            "per_category": a.per_category, "gate": a.gate or None, "gate_frac": a.gate_frac,
            "census": dict(sorted(census.items())),
            "rows": len(out), "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["census"], indent=1))


# ===================================================================================================== generate
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def generate(a) -> None:
    """Ask the server for every prompt not already answered; write request + response in capture layout."""
    rows = [json.loads(ln) for ln in Path(a.prompts).open() if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    root = Path(a.out)
    url = a.url.rstrip("/") + "/v1/chat/completions"
    lock = threading.Lock()
    done = Counter()

    def one(r):
        req_f = root / f"{r['id']}.json"
        resp_f = req_f.with_name(req_f.name[: -len(".json")] + ".response.json")
        if resp_f.exists() and json.loads(resp_f.read_text()).get("complete"):
            with lock:
                done["skipped"] += 1
            return
        body = {"model": a.model, "max_tokens": a.max_tokens, **SAMPLER, **r["body"], "stream": False}
        req_f.parent.mkdir(parents=True, exist_ok=True)
        req_f.write_text(json.dumps({"at": _now(), "url": url, "method": "POST", "sessionId": r["id"],
                                     "body": body}))
        rec = {"at": None, "status": None, "body": "", "complete": False, "error": None}
        try:
            q = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(q, timeout=a.timeout) as h:
                rec["status"], rec["body"] = h.status, h.read().decode()
            rec["complete"] = True
        except urllib.error.HTTPError as e:
            rec["status"], rec["body"], rec["error"] = e.code, e.read().decode(errors="replace"), str(e)
        except Exception as e:  # timeout / connection: recorded, retried on the next run
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["at"] = _now()
        resp_f.write_text(json.dumps(rec))
        finish = None
        if rec["complete"]:
            try:
                finish = json.loads(rec["body"])["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError):
                finish = "unparsed"
        with lock:
            done[f"finish:{finish}" if rec["complete"] else "error"] += 1
            n = sum(done.values())
            if n % a.log_every == 0 or n == len(rows):
                print(f"{_now()} {n}/{len(rows)} {dict(done)}", flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(one, rows))
    print(json.dumps(dict(done)))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate-ngrams")
    g.add_argument("--out", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--src", required=True, help="comma list of globs of RL-blend JSONL files")
    p.add_argument("--gate", default="", help="gate-ngrams output")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--gate-frac", type=float, default=0.1, help="share of a row's 13-grams in the gate that drops it")
    p.add_argument("--per-category", type=int, default=75)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--out", required=True)
    q = sub.add_parser("generate")
    q.add_argument("--prompts", required=True)
    q.add_argument("--url", required=True, help="server base URL (llama-server / opencoti)")
    q.add_argument("--model", required=True, help="served model name")
    q.add_argument("--out", required=True, help="capture dir (omk_qad.rollouts --capture)")
    q.add_argument("--max-tokens", type=int, default=16384)
    q.add_argument("--parallel", type=int, default=1, help="concurrent requests (match the server's -np)")
    q.add_argument("--timeout", type=float, default=3600)
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    {"gate-ngrams": gate_ngrams, "prompts": prompts, "generate": generate}[a.cmd](a)


if __name__ == "__main__":
    main()
