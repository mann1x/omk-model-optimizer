#!/usr/bin/env python3
"""bm-889 sft2: does a regenerated assistant turn pass? (one job from sft2_jobs.py + one model response)

A response is an OpenAI-style assistant message plus its finish reason: {"content", "reasoning_content",
"tool_calls", "finish_reason"}. Checks, in order (every failing check is reported, the first is the reason):

  finish     finish_reason is stop / tool_calls (a length stop is a runaway, never kept)
  reasoning  a native thought is present when the turn answers a USER message (after a tool result Gemma-4
             may continue without one; --allow-no-reasoning for gold runs
             of sources that carry none)
  loop       no sentence/line unit (>= MIN_UNIT chars) repeated RUN_REPEATS times back to back, or SHARE_REPEATS
             times covering SHARE_MIN of the text, in the reasoning or the text (sentence-level on purpose: char
             regexes are blind to sentence loops and fire on code indentation)
  foreign    no foreign surface token (an-finetune validate_replay_render FOREIGN) in the text or the reasoning
  call       type "call": >= 1 call; every name is a declared tool; arguments are an object; required keys present,
             JSON types match the schema; and the calls are EQUIVALENT to the recorded ones (same function names in
             order; hermes: same arguments after normalisation; other sources: the same values for the identifying
             keys — path / file / url / id / name / a one-word subcommand ... — free-text arguments (query,
             pattern, a whole shell command line) are only shape-checked),
             because the recorded tool result answers only the recorded call
  nocall     type "nocall": no call, non-empty text
  answer     type "final" / "text": no call (a call where the recorded turn answered is checked as a call against
             nothing — rejected), non-empty text; with a reference: MCQ letter / last \\boxed{} / entity must match

``--gold JOBS``: validates every job's RECORDED turn as if it were the response (a check the gold fails is a broken
check), plus the loop detector's positive / negative controls. Prints per source and type pass rates and reasons.
"""
import argparse
import json
import re
import sys
import threading
from collections import Counter

sys.path.insert(0, "/srv/ml/an-finetune/docker")
from validate_replay_render import FOREIGN  # noqa: E402  (an-finetune's foreign surface tokens)

# Gemma-4 control tokens: tokenized as special wherever they appear, so they are never valid generated text
CONTROL = ("<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|tool>", "<tool|>", "<|turn>", "<turn|>",
           "<|channel>", "<channel|>", '<|"|>', "<bos>", "<eos>", "<pad>")
RUN_REPEATS = 3
SHARE_REPEATS = 8
SHARE_MIN = 0.10
SCATTER_REPEATS = 3  # the same thought anywhere in a turn, back to back or not (bug-868)
SHINGLE_WORDS, SHINGLE_REPEATS = 12, 3
WAIT_MAX = 10
MIN_UNIT = 24
ID_KEYS = re.compile(r"(path|file|dir|command|cmd|url|uri|id|name|symbol|target)s?$", re.I)
# Free text is shape-checked only: a search query / pattern, and a command whose recorded value is a whole command
# line (a shell `command`), have many equivalent spellings. A one-word command (str_replace_editor `view`, a
# subcommand) is still identifying. (2026-10-30: exact-matching these rejected sensible bash calls — swe kept 1/55
# execute_bash jobs, opencode lost 574 jobs to `bash.command` alone — and biased the kept calls to verbatim copies.)
SUBCOMMAND_KEYS = re.compile(r"(command|cmd)s?$", re.I)


def _identifying(key: str, value) -> bool:
    if not ID_KEYS.search(key) or not isinstance(value, (str, int, float)) or len(str(value)) > 300:
        return False
    return not (SUBCOMMAND_KEYS.search(key) and isinstance(value, str) and len(value.split()) > 1)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
UNIT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _wordy(u: str) -> bool:
    """Mostly letters/digits/spaces: a `# ------` separator or a `}}}` run is layout, not a repeated thought."""
    return sum(ch.isalnum() or ch.isspace() for ch in u) >= 0.5 * len(u)


def loop_units(text: str) -> tuple[int, str]:
    """(max repeats, unit) of the most repeated sentence/line unit of at least MIN_UNIT chars."""
    c = Counter(u.strip() for u in UNIT_SPLIT.split(text or "") if len(u.strip()) >= MIN_UNIT and _wordy(u.strip()))
    if not c:
        return 0, ""
    u, n = c.most_common(1)[0]
    return n, u


def is_loop(text: str) -> tuple[bool, str]:
    """A loop: >= RUN_REPEATS identical units back to back, or one unit >= SHARE_REPEATS times AND >= SHARE_MIN
    of the text. Legitimate repetition is neither (a list where every item carries the same date, the same
    import in several files of a final answer: 2026-10-29 gold false positives, 16/2,420 recorded turns)."""
    units = [u.strip() for u in UNIT_SPLIT.split(text or "") if len(u.strip()) >= MIN_UNIT]
    run, best, prev, top = 0, 0, None, ""
    for u in units:
        if not _wordy(u):  # layout breaks a run; it must not join two sentences it separates
            run, prev = 0, None
            continue
        run = run + 1 if u == prev else 1
        prev = u
        if run > best:
            best, top = run, u
    if best >= RUN_REPEATS:
        return True, f"run {best}x {top[:60]!r}"
    n, u = loop_units(text)
    if n >= SHARE_REPEATS and n * len(u) >= SHARE_MIN * max(1, len(text)):
        return True, f"share {n}x {u[:60]!r}"
    return False, ""


def repetition(reason: str, text: str) -> str | None:
    """A thought the turn repeats anywhere, reasoning and text together, fenced code aside (bug-868): the same unit,
    the same unit but for digits, case and punctuation, or a run of SHINGLE_WORDS words. ``is_loop`` needs them back to
    back, or at 10 % of the text; a loop that scatters its repeats between other sentences passes it."""
    prose = FENCE.sub(" ", f"{reason}\n{text}")
    if len(re.findall(r"\bWait\b", prose)) >= WAIT_MAX:
        return "chatter:wait"
    us = [u.strip() for u in UNIT_SPLIT.split(prose) if len(u.strip()) >= MIN_UNIT and _wordy(u.strip())]
    near = [n for n in (re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", re.sub(r"\d+", "#", u.lower()))).strip() for u in us)
            if len(n) >= MIN_UNIT]
    for kind, units in (("scatter", us), ("near", near)):
        c = Counter(units).most_common(1)
        if c and c[0][1] >= SCATTER_REPEATS:
            return f"loop:{kind} {c[0][1]}x {c[0][0][:60]!r}"
    words = re.findall(r"\w+", prose.lower())
    c = Counter(tuple(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)).most_common(1)
    if c and c[0][1] >= SHINGLE_REPEATS:
        return f"loop:shingle {c[0][1]}x {' '.join(c[0][0])[:60]!r}"
    return None


def _args(call) -> dict | None:
    fn = call.get("function", call)
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a or "{}")
        except json.JSONDecodeError:
            return None
    return a if isinstance(a, dict) else None


def _name(call) -> str:
    return (call.get("function", call) or {}).get("name") or ""


def _schema(tools, name) -> dict | None:
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            return fn.get("parameters") or {}
    return None


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def check_calls(job, calls) -> list[str]:
    bad = []
    rec = job["recorded"].get("tool_calls") or []
    if [_name(c) for c in calls] != [_name(c) for c in rec]:
        bad.append(f"call:names {[_name(c) for c in calls]} != recorded {[_name(c) for c in rec]}")
    for c, r in zip(calls, rec):
        name, a, ra = _name(c), _args(c), _args(r) or {}
        sch = _schema(job["tools"], name)
        if sch is None:
            bad.append(f"call:undeclared {name}")
            continue
        if a is None:
            bad.append(f"call:args_not_object {name}")
            continue
        props = (sch or {}).get("properties") or {}
        for k in (sch or {}).get("required") or []:
            if k not in a:
                bad.append(f"call:missing_required {name}.{k}")
        for k, v in a.items():
            t = (props.get(k) or {}).get("type")
            if isinstance(t, str) and t in JSON_TYPES and v is not None and not isinstance(v, JSON_TYPES[t]):
                bad.append(f"call:type {name}.{k} {type(v).__name__}!={t}")
            if t in ("integer", "number") and isinstance(v, bool):
                bad.append(f"call:type {name}.{k} bool!={t}")
        if job["source"] == "hermes":
            if _norm(a) != _norm(ra):
                bad.append(f"call:args_differ {name}")
        else:
            for k, v in ra.items():
                if _identifying(k, v):
                    if _norm(a.get(k)) != _norm(v):
                        bad.append(f"call:id_arg_differs {name}.{k}")
    return bad


def last_boxed(text: str) -> str | None:
    """The content of the last ``\\boxed{...}``, braces matched at any depth (the BOXED regex stops at two)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k].strip()
    return None


LATEX_NOISE = re.compile(r"\\left|\\right|\\[,;:! ]|\\displaystyle|\\text\{\s*\}|[\s{}$]")


def same_math(a: str, b: str) -> bool:
    """Equal after LaTeX layout normalisation (spacing, \\left/\\right, grouping braces), else math_verify."""
    if LATEX_NOISE.sub("", a).lower() == LATEX_NOISE.sub("", b).lower():
        return True
    # math_verify's timeouts use signals, which work only on the main thread: in sft2_generate's worker threads they
    # made verify() return False for correct answers (2026-10-30: 8 math finals rejected live, equal when rescored).
    t = 5 if threading.current_thread() is threading.main_thread() else None
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${b}$", parsing_timeout=t), parse(f"${a}$", parsing_timeout=t), timeout_seconds=t))
    except Exception:  # noqa: BLE001  (an unparsable answer is not equal)
        return False


def check_answer(ref, text: str) -> list[str]:
    if not ref:
        return []
    if ref["kind"] == "mcq":
        got = last_boxed(text or "")
        return [] if got is not None and _norm(got) == _norm(ref["value"]) else [f"answer:mcq {got!r}"]
    if ref["kind"] == "boxed":
        got = last_boxed(text or "")
        return [] if got is not None and same_math(got, ref["value"]) else [f"answer:boxed {got!r}"]
    if ref["kind"] == "entity":
        return [] if _norm(str(ref["value"])) in _norm(text or "") else ["answer:entity_missing"]
    return []


def context_defect(context: list[dict]) -> str | None:
    """A history call without its answer: the template renders a never-closed ``<|tool_response>`` (no answer) or
    one response for several calls (partial)."""
    for i, m in enumerate(context):
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = i + 1
            while k < len(context) and context[k]["role"] == "tool":
                k += 1
            if k == i + 1:
                return "context:unanswered_call"
            if k - i - 1 < len(m["tool_calls"]):
                return "context:partial_answer"
    return None


def validate(job: dict, resp: dict, allow_no_reasoning: bool = False) -> list[str]:
    """Reasons the response fails (empty: it passes)."""
    bad = []
    defect = context_defect(job.get("context") or [])
    if defect:
        bad.append(defect)
    fin = resp.get("finish_reason")
    if fin not in ("stop", "tool_calls"):
        bad.append(f"finish:{fin}")
    if fin == "tool_calls" and not resp.get("tool_calls"):  # the parser stripped a call it could not read
        bad.append("call:parse_lost")
    text, reason = resp.get("content") or "", resp.get("reasoning_content") or ""
    # Gemma-4 thinks before answering a USER message; after a tool result it often continues the same turn
    # without a thought (smoke gen_smoke1 2026-10-30: hermes final, agentic text, opencode call turns, all
    # well-formed) — that is its own behaviour, not a defect, so a thought is required only after a user turn.
    after_user = (job.get("context") or [{}])[-1].get("role") == "user"
    if not reason.strip() and after_user and not allow_no_reasoning:
        bad.append("reasoning:missing")
    for where, s in (("reasoning", reason), ("text", text)):
        looped, how = is_loop(s)
        if looped:
            bad.append(f"loop:{where} {how}")
        if any(f in s for f in FOREIGN):
            bad.append(f"foreign:{where}")
        if any(c in s for c in CONTROL):  # a parse the server left in the text (a failed call, a leaked channel tag)
            bad.append(f"control:{where}")
    why = repetition(reason, text)
    if why:
        bad.append(why)
    calls = resp.get("tool_calls") or []
    if calls and text.strip():  # the template puts tool_calls first: the target would be text AFTER <tool_call|> (bug-866)
        bad.append("call:text_beside")
    if any(c in json.dumps(calls, ensure_ascii=False) for c in CONTROL):
        bad.append("control:args")
    t = job["type"]
    if t == "call":
        bad += ["call:none"] if not calls else check_calls(job, calls)
    elif t == "nocall":
        bad += (["nocall:called"] if calls else []) + ([] if text.strip() else ["nocall:empty"])
    else:
        bad += (["answer:called"] if calls else []) + ([] if text.strip() else ["answer:empty"])
        bad += check_answer(job.get("reference"), text)
    return bad


def gold(jobs_path: str) -> None:
    pos = "Let me look at the draw call again. " + "dDec(c,x){this.dc.forEach(d=>{if(d.tp==='stl'){c.fillRect(d.x-x)}})}\n" * 6
    neg = "\n".join(["    }", "  }", "}", "    return x;", "    }", "  }", "}"] * 5)  # code indentation, short units
    alt = ("I should fix the dDec draw call for the steel tiles now.\nLet me re-read lines 88 to 92 of the file again.\n"
           * 10)  # a two-unit loop: no unit ever repeats back to back
    items = "\n".join(f"- Signal {k}: price crossed the 50-day average\n- Date of Signal: 2024-06-28" for k in range(5))
    print("CONTROL positive (6x code line back to back):", is_loop(pos)[0])
    print("CONTROL positive (two-unit alternating loop x10):", is_loop(alt)[0])
    print("CONTROL negative (brace indentation x5):", not is_loop(neg)[0])
    print("CONTROL negative (5 list items sharing a date line):", not is_loop(items)[0])
    sep = "\n".join(f"# {'-' * 58}\ndef f{k}(x):\n    return x + {k}" for k in range(12))
    print("CONTROL negative (12 separator comments between functions):", not is_loop(sep)[0])
    filler = "".join(f"Then step {w} follows from the previous one by {w} direct argument. " for w in "abcdefgh")
    thought = "So the integral must be evaluated again from the start. "
    print("CONTROL positive (sentence x3, scattered):", bool(repetition(f"{thought}{filler}{thought}{filler}{thought}", "")))
    print("CONTROL positive (same sentence but its number x3):", bool(repetition(
        " ".join(f"Let me recompute the sum for row {k} of the table. {filler}" for k in (3, 7, 9)), "")))
    print("CONTROL positive (12-word run x3, tails differ):", bool(repetition(" ".join(
        f"we still need to check whether the value of the second term is {w} here." for w in ("ok", "two", "x")), "")))
    print("CONTROL positive (Wait x10):", bool(repetition(" ".join(f"Wait, is {w} the right term here?" for w in "abcdefghij"), "")))
    print("CONTROL negative (sentence x2):", not repetition(f"{thought}{filler}{thought}", ""))
    print("CONTROL negative (fenced code x5):", not repetition("```\n" + "total = total + value_of_the_item(i)\n" * 5 + "```", ""))
    print("CONTROL negative (Wait x9):", not repetition(" ".join(f"Wait, is {w} the right term here?" for w in "abcdefghi"), ""))
    tally, reasons = Counter(), Counter()
    for line in open(jobs_path):
        j = json.loads(line)
        r = dict(j["recorded"])
        r["finish_reason"] = "tool_calls" if r.get("tool_calls") else "stop"
        bad = validate(j, r, allow_no_reasoning=True)
        key = f"{j['source']}/{j['type']}"
        tally[key, bool(bad)] += 1
        for b in bad:
            reasons[f"{key} {b.split(' ')[0]}"] += 1
    for key in sorted({k for k, _ in tally}):
        ok, ko = tally[key, False], tally[key, True]
        print(f"GOLD {key:22s} pass {ok}/{ok + ko} ({100 * ok / max(1, ok + ko):.1f}%)")
    for r, n in reasons.most_common(25):
        print(f"  fail {n:5d} {r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", help="jobs.jsonl: validate the recorded turns + loop controls")
    a = ap.parse_args()
    if a.gold:
        gold(a.gold)
    else:
        ap.print_help()  # library use: from sft2_validate import validate
