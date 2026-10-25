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
               ``.response.json``; resumable (a complete response is never re-asked), ``--parallel`` requests,
               ``--redo-capped`` re-asks rows capped below a raised ``--max-tokens``; atomic writes, one process
               per capture dir (``<out>.lock``), progress in ``<out>.state.json``
  control      pause | resume | drain | status of a running generate (also SIGUSR2 = pause/resume toggle,
               SIGUSR1 / SIGTERM = drain: finish the in-flight request, then exit; a 2nd SIGTERM stops at once)

    python -m omk_qad.replay gate-ngrams --out gate13.json
    python -m omk_qad.replay prompts --src 'RL-*/rlvr*.jsonl' --gate gate13.json --per-category 75 --out p.jsonl
    python -m omk_qad.replay generate --prompts p.jsonl --url http://host:port --model NAME --out CAPDIR
    python -m omk_qad.replay control --out CAPDIR drain    # then restart, e.g. with a higher --max-tokens --redo-capped
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .runctl import Control, RunLock, atomic_write, check_identity, file_id, parse_allow

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
                # a masked row (RL-Ultra DAPO/Skywork: _hf_question_placeholder, question "") has an empty user
                # turn until fill_placeholders.py restores it; the model would answer "" (bug-843)
                if not (body["messages"][-1].get("content") or "").strip():
                    census[f"{cat}:empty_user_turn"] += 1
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


def _capped_tokens(rec: dict) -> int | None:
    """completion_tokens of a complete response that stopped on the length cap, else None."""
    try:
        b = json.loads(rec["body"])
        if b["choices"][0].get("finish_reason") == "length":
            return int(b.get("usage", {}).get("completion_tokens") or 0)
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    return None


def _side(out: Path, kind: str) -> Path:
    """Sibling of the capture dir (never inside it: rollouts --capture scans it recursively)."""
    return out.with_name(f"{out.name}.{kind}")


def generate(a) -> None:
    """Ask the server for every prompt not already answered; write request + response in capture layout.

    Resumable: a complete response is never re-asked; a request without a response, or a response file that does
    not parse (moved to ``<out>.bad/``), is asked again. ``--redo-capped``: a complete response that hit a LOWER
    length cap than ``--max-tokens`` counts as not answered; its pair is moved (never overwritten) to
    ``<out>.capped/<old cap>/`` and the prompt is re-asked. Control: ``replay control`` / signals (runctl.Control).
    One process per capture dir (``<out>.lock``); progress in ``<out>.state.json``; ``<out>.identity.json`` refuses
    a restart with another served model, sampler or prompts file unless ``--allow-change`` names it (the url is a
    location, not identity: it is recorded in the state file) (``--max-tokens``
    is not identity: a raised cap is what ``--redo-capped`` is for).
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    rows = [json.loads(ln) for ln in Path(a.prompts).open() if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(_side(root, "lock"), "replay generate")
    check_identity(_side(root, "identity.json"),
                   {"model": a.model, "sampler": SAMPLER, "prompts": file_id(a.prompts), "limit": a.limit},
                   parse_allow(a.allow_change), what="replay generate")
    atomic_write(_side(root, "pid"), f"{os.getpid()}\n")
    ctl = Control(_side(root, "control"), name="replay generate")
    url = a.url.rstrip("/") + "/v1/chat/completions"
    lock = threading.Lock()
    done, redo = Counter(), Counter()
    inflight_ids: set[str] = set()
    status = {"v": "running"}

    def state() -> None:
        with lock:
            snap = {"pid": os.getpid(), "status": status["v"], "at": _now(), "url": url, "max_tokens": a.max_tokens,
                    "redo_capped": a.redo_capped, "rows": len(rows), "counts": {**done, **redo},
                    "inflight": sorted(inflight_ids)}
        atomic_write(_side(root, "state.json"), json.dumps(snap, indent=1))

    def move_aside(src: Path, dst_dir: Path, rid: str, suf: str) -> None:
        tgt = dst_dir / f"{rid}{suf}"
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.exists():
            raise SystemExit(f"refusing to overwrite archived {tgt}")
        os.replace(src, tgt)

    def one(r):
        req_f = root / f"{r['id']}.json"
        resp_f = req_f.with_name(req_f.name[: -len(".json")] + ".response.json")
        if resp_f.exists():
            try:
                old = json.loads(resp_f.read_text())
            except ValueError:
                move_aside(resp_f, _side(root, "bad"), r["id"], ".response.json")
                old = None
                with lock:
                    redo["unparsable_moved"] += 1
            if old and old.get("complete"):
                cap = _capped_tokens(old) if a.redo_capped else None
                if cap is None or cap >= a.max_tokens:
                    with lock:
                        done["skipped"] += 1
                    return
                for src, suf in ((req_f, ".json"), (resp_f, ".response.json")):
                    if src.exists():
                        move_aside(src, _side(root, "capped") / str(cap), r["id"], suf)
                with lock:
                    redo[f"redo_capped_{cap}"] += 1
        body = {"model": a.model, "max_tokens": a.max_tokens, **SAMPLER, **r["body"], "stream": False}
        req_f.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(req_f, json.dumps({"at": _now(), "url": url, "method": "POST", "sessionId": r["id"],
                                         "body": body}))
        with lock:
            inflight_ids.add(r["id"])
        state()
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
        atomic_write(resp_f, json.dumps(rec))
        finish = None
        if rec["complete"]:
            try:
                finish = json.loads(rec["body"])["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError):
                finish = "unparsed"
        with lock:
            inflight_ids.discard(r["id"])
            done[f"finish:{finish}" if rec["complete"] else "error"] += 1
            n = sum(done.values())
            if n % a.log_every == 0 or n == len(rows):
                print(f"{_now()} {n}/{len(rows)} {dict(done)}", flush=True)
        state()

    pending, futs, exhausted, last_mode = iter(rows), set(), False, "run"
    try:
        with ThreadPoolExecutor(a.parallel) as ex:
            while True:
                mode = ctl.mode()
                if mode != last_mode:
                    print(f"{_now()} mode {last_mode} -> {mode} ({len(futs)} in flight finish first)", flush=True)
                    last_mode = mode
                while mode == "run" and not exhausted and len(futs) < a.parallel:
                    r = next(pending, None)
                    if r is None:
                        exhausted = True
                    else:
                        futs.add(ex.submit(one, r))
                if not futs:
                    if exhausted:
                        status["v"] = "done"
                        break
                    if mode == "drain":
                        status["v"] = "drained"
                        break
                    if status["v"] != "paused":
                        status["v"] = "paused"
                        print(f"{_now()} PAUSED (replay control --out {root} resume, or SIGUSR2)", flush=True)
                        state()
                    time.sleep(2)
                    continue
                if status["v"] == "paused":
                    status["v"] = "running"
                fin, futs = wait(futs, timeout=2, return_when=FIRST_COMPLETED)
                for f in fin:
                    f.result()
    except BaseException as e:
        status["v"] = "failed"
        state()
        print(f"GENERATE_FAILED {_now()} {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        ctl.restore()
        run_lock.release()
    state()
    print(json.dumps({**dict(done), **dict(redo)}))
    print(f"GENERATE_{status['v'].upper()} {_now()}", flush=True)


def control(a) -> None:
    """Set the run mode of the generate process on ``--out`` (or just print its state)."""
    root = Path(a.out)
    if a.action != "status":
        atomic_write(_side(root, "control"), {"resume": "run"}.get(a.action, a.action) + "\n")
        print(f"{_side(root, 'control')} <- {a.action}")
    sf = _side(root, "state.json")
    print(sf.read_text() if sf.exists() else f"no {sf}")


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
    q.add_argument("--allow-change", nargs="*", default=[], metavar="KEY[=VALUE]",
                   help="identity keys (model sampler prompts limit) allowed to differ from the capture")
    q.add_argument("--redo-capped", action="store_true",
                   help="re-ask complete responses capped below --max-tokens (the old pair is archived)")
    c = sub.add_parser("control", help="pause / resume / drain a running generate, or print its state")
    c.add_argument("--out", required=True, help="the capture dir of the running generate")
    c.add_argument("action", choices=("pause", "resume", "drain", "status"))
    a = ap.parse_args()
    {"gate-ngrams": gate_ngrams, "prompts": prompts, "generate": generate, "control": control}[a.cmd](a)


if __name__ == "__main__":
    main()
