#!/usr/bin/env python3
"""bm-889 sft2: regenerate each job's assistant turn with Gemma-4 itself, validate, keep the first passing sample.

For every job (sft2_jobs.py) up to ``--samples`` attempts are sent to an OpenAI-compatible chat endpoint serving
the untouched base (llama-server :8261 or vLLM) with the job's recorded history, its tools, thinking on and the
Gemma default sampler; each response is checked by sft2_validate.validate. The first passing response is kept
(never the shortest: no length bias). Every attempt is logged with its reasons, so the pass rate per source and
turn type is itself a measurement of the base's tool-call degeneration.

Before the first job the server is checked: /props model_path must be ``--expect-model`` and its chat template
sha256 (without the trailing newline) ``--expect-template`` — a generator that is not the base is refused.

Out (OUT dir): attempts.jsonl (one line per attempt: job, attempt, ok, reasons, finish, tokens, seconds),
kept.jsonl (one line per job that passed: job, source, type, ref, turn, context, tools, response),
GEN_OK / GEN_FAIL in the log. Resumable: jobs already in attempts.jsonl with a kept sample or all attempts used
are skipped.

Jobs from sft2_plen.py carry ``prompt_tokens``: their requests are capped at ``--window`` - prompt_tokens, so
every kept turn fits the training window (a length cap hit is a rejected attempt, finish ``length``).

Usage: sft2_generate.py JOBS OUT --url http://172.17.0.1:8261 [--samples 4] [--limit N] [--workers 1]
"""
import argparse
import hashlib
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sft2_validate import repetition, validate

MODEL = None  # vLLM: the served model id (requests name it); llama-server: unused
WINDOW = 16384  # the training --seq (as sft1); set by --window
CHECK_CHARS = 2000
SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0}  # Gemma-4 default, as the gate's arm sends
SAMPLER_NAME = "gemma-4-default"  # v9 penalties A/B on known-looping prompts: no gain, tool pass 88 % -> 80 % (ab_loop.jsonl)


def post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(url: str, body: dict, timeout: int) -> tuple[dict, dict]:
    """A turn with no call to come, streamed and dropped the moment it can no longer pass: a loop runs on to the token
    cap otherwise, and 73 % of the refill's tokens went to rejected attempts (repetition() first fires at a third of a
    failing turn, a fifth on math, science and cp). Closing the socket aborts the request on the server."""
    req = urllib.request.Request(url, data=json.dumps(body | {"stream": True, "stream_options": {"include_usage": True}}).encode(),
                                 headers={"Content-Type": "application/json"})
    reason, text, fin, usage, seen = "", "", None, {}, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data: ") or raw.startswith(b"data: [DONE]"):
                continue
            ev = json.loads(raw[6:])
            usage = ev.get("usage") or usage
            if not ev.get("choices"):
                continue
            ch = ev["choices"][0]
            fin = ch.get("finish_reason") or fin
            d = ch.get("delta") or {}
            reason += d.get("reasoning") or d.get("reasoning_content") or ""
            text += d.get("content") or ""
            if d.get("tool_calls"):
                fin = "abort:called"
                break
            if len(reason) + len(text) - seen >= CHECK_CHARS:
                seen = len(reason) + len(text)
                why = repetition(reason, text)
                if why:
                    fin = f"abort:{why.split(' ')[0]}"
                    break
    return {k: v for k, v in (("role", "assistant"), ("reasoning_content", reason), ("content", text),
                              ("finish_reason", fin)) if v}, usage


def norm_message(m: dict) -> dict:
    """The response as the trainer reads it: vLLM 0.20 names the thought ``reasoning`` (llama-server:
    ``reasoning_content``) and adds empty OpenAI fields (refusal, annotations, audio, function_call)."""
    m = {k: v for k, v in m.items() if v is not None and k not in ("refusal", "annotations", "audio", "function_call")}
    if "reasoning" in m:
        r = m.pop("reasoning")
        m.setdefault("reasoning_content", r)
    return m


def wire(messages: list[dict]) -> list[dict]:
    """vLLM takes the OpenAI schema strictly: tool-call arguments as a JSON string (llama-server also takes a dict);
    it parses them back for the template, so the render is unchanged (check_vllm proves it token by token)."""
    if MODEL is None:
        return messages
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = m | {"tool_calls": [c | {"function": c["function"] | {"arguments": json.dumps(
                c["function"]["arguments"], ensure_ascii=False)}} if not isinstance(c["function"].get("arguments"),
                str) else c for c in m["tool_calls"]]}
        out.append(m)
    return out


def check_vllm(url: str, hf_dir: str, template_file: str, job: dict) -> dict:
    """A vLLM server has no /props: its model must be the base HF dir, and its render of a real job (tools and all)
    must be token-identical to the local render with the gate's template file."""
    from transformers import AutoTokenizer

    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as r:
        m = json.loads(r.read())["data"][0]
    if m.get("root") != hf_dir:
        raise SystemExit(f"GEN_FAIL vLLM model {m.get('root')} != {hf_dir}")
    global MODEL
    MODEL = m["id"]
    tpl = open(template_file).read()
    sha = hashlib.sha256(tpl.rstrip("\n").encode()).hexdigest()
    ctx = json.loads(json.dumps(job["context"]))
    for msg in ctx:  # the template renders dict arguments (omk_qad.rollouts._norm)
        for c in msg.get("tool_calls") or []:
            if isinstance(c["function"].get("arguments"), str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"] or "{}")
    local = AutoTokenizer.from_pretrained(hf_dir).apply_chat_template(
        ctx, tools=job["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=True, chat_template=tpl)
    body = {"model": m["id"], "messages": wire(job["context"]), "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True}, **({"tools": job["tools"]} if job["tools"] else {})}
    served = post(f"{url}/tokenize", body, 60)["tokens"]
    ids = AutoTokenizer.from_pretrained(hf_dir)(local, add_special_tokens=False)["input_ids"]
    if served != ids:
        raise SystemExit(f"GEN_FAIL vLLM render of {job['job']} differs from the local render ({len(served)} vs "
                         f"{len(ids)} tokens)")
    return {"engine": "vllm", "model_root": m["root"], "template_sha256": sha, "render_probe": job["job"],
            "render_tokens": len(ids)}


def check_server(url: str, model: str, template: str) -> dict:
    with urllib.request.urlopen(f"{url}/props", timeout=30) as r:
        p = json.loads(r.read())
    sha = hashlib.sha256((p.get("chat_template") or "").rstrip("\n").encode()).hexdigest()
    if model and p.get("model_path") != model:
        raise SystemExit(f"GEN_FAIL server model {p.get('model_path')} != {model}")
    if template and not sha.startswith(template):
        raise SystemExit(f"GEN_FAIL server template {sha[:16]} != {template}")
    return {"model_path": p.get("model_path"), "template_sha256": sha}


def attempt(url: str, job: dict, k: int, max_tokens: int, timeout: int, seed: int) -> tuple[dict, dict]:
    if job.get("prompt_tokens"):  # sft2_plen.py: the turn must fit the training window with its prompt
        # 32 tokens of margin: vLLM counts one prompt token more than the local render (HTTP 400 at exactly the
        # window, gen_smoke2_vllm 2026-10-30), and the template's turn close follows the generation
        max_tokens = min(max_tokens, WINDOW - job["prompt_tokens"] - 32)
    body = {"messages": wire(job["context"]), "tools": job["tools"], "max_tokens": max_tokens, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": True}, **SAMPLER, **({"model": MODEL} if MODEL else {})}
    if not job["tools"]:
        body.pop("tools")
    t0 = time.time()
    try:
        if job["tools"]:  # the streaming tool parser is unproven: 60 live pairs lost reasoning on tool jobs, none without tools
            r = post(f"{url}/v1/chat/completions", body, timeout)
            ch = r["choices"][0]
            msg = norm_message(ch["message"])
            msg["finish_reason"] = ch.get("finish_reason")
            usage = r.get("usage") or {}
        else:
            msg, usage = stream(f"{url}/v1/chat/completions", body, timeout)
    except Exception as e:  # noqa: BLE001  (a failed request is a failed attempt, logged with its error)
        msg, usage = {"finish_reason": f"error:{type(e).__name__}"}, {}
    reasons = validate(job, msg)
    log = {"job": job["job"], "source": job["source"], "type": job["type"], "attempt": k, "ok": not reasons, "sampler": SAMPLER_NAME,
           "reasons": reasons, "finish": msg.get("finish_reason"), "completion_tokens": usage.get("completion_tokens"),
           "prompt_tokens": usage.get("prompt_tokens"), "seconds": round(time.time() - t0, 1),
           **({} if not reasons else {"response": msg})}  # a rejected response is kept for diagnosis
    return log, msg


def run_job(a, job: dict) -> list[tuple[dict, dict]]:
    out = []
    rng = random.Random(f"{a.seed}/{job['job']}")
    for k in range(a.samples):
        if a.quota.get(job["source"], 1) <= 0:  # refilled: the job stays unattempted
            break
        log, msg = attempt(a.url, job, k, a.max_tokens, a.timeout, rng.randrange(1 << 31))
        out.append((log, msg))
        if log["ok"]:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--url", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N jobs of a seeded shuffle (0: all)")
    ap.add_argument("--workers", type=int, default=1, help="requests in flight (match the server's slots)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--window", type=int, default=16384, help="training window; jobs with prompt_tokens get "
                    "max_tokens = min(--max-tokens, window - prompt_tokens)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=889)
    ap.add_argument("--expect-model", default="/mnt/sdc/opencoti-models/gemma-4-26B-A4B/it-bf16/gemma-4-26B-A4B-it-BF16-eos106.gguf")
    ap.add_argument("--expect-template", default="6a1015c47ccfcfa6")
    ap.add_argument("--engine", choices=["llama", "vllm"], default="llama", help="vllm: identity by /v1/models root "
                    "+ a token-identical /tokenize render of the first job with --template-file")
    ap.add_argument("--expect-hf", default="/mnt/sdc/ml/models/gemma-4-26B-A4B-it")
    ap.add_argument("--template-file", default="/mnt/sdc/omk-model-optimizer/g4_recovery/sft2_build/"
                    "served_template_6a1015c4.jinja")
    ap.add_argument("--probe", help="job for the vLLM render check (default: first job with tools and a tool turn)")
    ap.add_argument("--quota", help='JSON {"source": turns}: stop a source once it has kept this many')
    a = ap.parse_args()
    a.quota = json.loads(a.quota) if a.quota else {}
    global WINDOW
    WINDOW = a.window
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in open(a.jobs)]
    random.Random(f"{a.seed}/order").shuffle(jobs)
    if a.engine == "vllm":
        probe = next(j for j in jobs if (j["job"] == a.probe if a.probe else j["tools"] and any(m["role"] == "tool" for m in j["context"])))
        ident = check_vllm(a.url, a.expect_hf, a.template_file, probe)
        if not ident["template_sha256"].startswith(a.expect_template):
            raise SystemExit(f"GEN_FAIL template file {ident['template_sha256'][:16]} != {a.expect_template}")
    else:
        ident = check_server(a.url, a.expect_model, a.expect_template)
    print("GEN_SERVER", json.dumps(ident), flush=True)
    if a.limit:
        jobs = jobs[: a.limit]
    done = set()
    att_p, kept_p = out / "attempts.jsonl", out / "kept.jsonl"
    if att_p.exists():
        per: dict = {}
        for line in open(att_p):
            r = json.loads(line)
            per.setdefault(r["job"], []).append(r)
        done = {j for j, rs in per.items() if any(r["ok"] for r in rs) or len(rs) >= a.samples}
    todo = [j for j in jobs if j["job"] not in done]
    if a.quota and kept_p.exists():
        for line in open(kept_p):
            r = json.loads(line)
            a.quota[r["source"]] = a.quota.get(r["source"], 0) - 1
    print(f"GEN_START jobs {len(jobs)} todo {len(todo)} samples {a.samples} workers {a.workers}", flush=True)
    t0, n_ok, n_att = time.time(), 0, 0
    with open(att_p, "a") as fa, open(kept_p, "a") as fk, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(lambda j: (j, run_job(a, j)), j) for j in todo]
        for i, fut in enumerate(as_completed(futs)):  # written as each job finishes: a crash loses only in-flight jobs
            job, tries = fut.result()
            for log, msg in tries:
                fa.write(json.dumps(log) + "\n")
                n_att += 1
                if log["ok"]:
                    n_ok += 1
                    if job["source"] in a.quota:
                        a.quota[job["source"]] -= 1
                    fk.write(json.dumps({k: job[k] for k in ("job", "source", "type", "ref", "turn", "context",
                                                               "tools", "recorded")} | {"response": msg}) + "\n")
            fa.flush()
            fk.flush()
            if (i + 1) % 10 == 0:
                print(f"GEN_PROGRESS {i + 1}/{len(todo)} kept {n_ok} attempts {n_att} "
                      f"{(time.time() - t0) / (i + 1):.1f} s/job", flush=True)
    print("GEN_QUOTA_LEFT", json.dumps({k: round(v) for k, v in a.quota.items()}), flush=True)
    print(f"GEN_OK jobs {len(todo)} kept {n_ok} attempts {n_att} seconds {round(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()
