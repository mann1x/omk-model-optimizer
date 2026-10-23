# SPDX-License-Identifier: Apache-2.0
"""Agentic rollouts -> fixed-length token windows with a training role per token (bf16 recovery data).

Input: JSONL rows ``{"id", "messages", "tools"?}`` in OpenAI chat form (assistant ``reasoning_content``,
``content``, ``tool_calls``; ``tool`` results). An assistant message may carry ``loop_spans``:
``[{"field": "reasoning_content" | "content", "start": c0, "end": c1}]`` (character offsets in that field,
e.g. from automaticum's detectors).

Each token gets a role (what the trainer does at the position that PREDICTS it):

  0 ignore   prompt, system, user, injected tool responses, template text the model did not generate
  1 teacher  generated reasoning, prose, tool-call structure and the stop token (Gemma-4: ``<|tool_response>``
             ends a tool-calling turn, ``<turn|>`` a final answer) -> KL to the (stable) teacher
  2 anchor   generated code: string values of code-carrying tool arguments (``--code-args``) and the inside
             of fenced code blocks in prose -> KL to the original student (keeps its coding)
  3 loop     generated text inside a loop span -> teacher KL plus unlikelihood of the repeated token

Rendering is the model's own chat template, so the context of every scored token is what the model saw when
it generated it. Templates that drop the reasoning of turns before the last user message (Gemma-4) are
handled by splitting a conversation into segments at each later user message: a segment is rendered alone,
its assistant turns after its last user message are scored, earlier ones are context. Every scored turn's
prompt must be an exact prefix of its segment's render, or the conversation is refused (counted, not
guessed).

A segment longer than ``--seq`` becomes overlapping windows ``[e - seq, e)`` stepped by ``--stride``; each
window scores only tokens no earlier window scored, so every scored token has at least ``seq - stride``
tokens of context (or the whole prefix) and is scored once.

    python -m omk_qad.rollouts --model DIR --in conv.jsonl --out X.roles.jsonl --seq 16384 [--stride 8192]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

IGNORE, TEACHER, ANCHOR, LOOP = 0, 1, 2, 3
PRIORITY = np.array([0, 2, 1, 3])  # a token spanning several roles takes the highest: loop > teacher > anchor
CODE_ARGS = ("content", "new_string", "old_string", "new_str", "old_str", "code", "patch", "diff", "file_text",
             "replace", "search", "text")
FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


class NotPrefixError(ValueError):
    """A scored turn's prompt is not a prefix of its segment's render (template not prefix-stable here)."""


def _norm(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as dicts (the template renders dicts; OpenAI wire form carries JSON strings)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def segment_ends(messages: list[dict]) -> list[int]:
    """End index (exclusive) of each segment: a new segment starts at a user message after a model turn."""
    ends, seen_model = [], False
    for j, m in enumerate(messages):
        if m["role"] == "user" and seen_model:
            ends.append(j)
            seen_model = False
        elif m["role"] in ("assistant", "tool"):
            seen_model = True
    return [*ends, len(messages)] if seen_model or not ends else ends


def render_segment(tok, messages: list[dict], tools, end: int, code_args=CODE_ARGS, stats=None):
    """(text, char roles) of ``messages[:end]``; the assistant turns after its last user message are scored."""
    kw = {"tools": tools, "tokenize": False, "enable_thinking": True}
    msgs = messages[:end]
    text = tok.apply_chat_template(msgs, **kw)
    roles = np.zeros(len(text), np.int8)
    last_user = max((j for j, m in enumerate(msgs) if m["role"] == "user"), default=-1)
    for i in range(last_user + 1, end):
        m = msgs[i]
        if m["role"] != "assistant":
            continue
        p = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, **kw)  # what the server sent
        a = tok.apply_chat_template(msgs[: i + 1], **kw)
        if not (text.startswith(p) and text.startswith(a) and len(a) > len(p)):
            raise NotPrefixError(f"turn {i}: prompt/turn render is not a prefix of the segment render")
        _label_turn(text, roles, len(p), len(a), m, code_args, stats)
    return text, roles


def _context_without_reasoning(messages: list[dict]) -> list[dict]:
    """``strip``: every earlier assistant turn without its reasoning (what such a provider sends back)."""
    return [{k: v for k, v in q.items() if k != "reasoning_content"} if q["role"] == "assistant" else q
            for q in messages]


def _context_last_inline(messages: list[dict]) -> list[dict]:
    r"""``last-inline``: Cerebriline SDK >= 4.100.260 when the template drops the reasoning field (ai-sdk.ts).

    Only the newest message that carries reasoning keeps it, trimmed and folded into its text as
    ``<think>R</think>\n`` + text (``<think>R</think>`` alone when it has no text); every other reasoning is
    dropped, and an assistant turn emptied by that is omitted. A ``pinned_reasoning`` turn (the harness's
    post-compaction retrospective) keeps its reasoning as text, ``R\n\n`` + text.
    """
    last = max((j for j, q in enumerate(messages) if q["role"] == "assistant" and q.get("reasoning_content")),
               default=-1)
    out = []
    for j, q in enumerate(messages):
        if q["role"] != "assistant":
            out.append(q)
            continue
        r = (q.get("reasoning_content") or "").strip()
        m = {k: v for k, v in q.items() if k != "reasoning_content"}
        text = m.get("content") or ""
        if r and q.get("pinned_reasoning"):
            m["content"] = f"{r}\n\n{text}"
        elif r and j == last:
            m["content"] = f"<think>{r}</think>\n{text}" if text else f"<think>{r}</think>"
        elif q.get("reasoning_content") and not text and not m.get("tool_calls"):
            continue  # emptied only by its dropped reasoning: the SDK sends no such turn
        out.append(m)
    return out


HISTORY_CONTEXT = {"strip": _context_without_reasoning, "last-inline": _context_last_inline}


def render_turns(tok, messages: list[dict], tools, history: str, code_args=CODE_ARGS, stats=None):
    """One (text, char roles) per assistant turn, for clients whose re-sent history changes every turn.

    The context of turn i is every earlier message as the client re-sent it (``HISTORY_CONTEXT[history]``),
    followed by turn i as generated, reasoning included; only turn i is scored.
    """
    kw = {"tools": tools, "tokenize": False, "enable_thinking": True}
    out = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ctx = HISTORY_CONTEXT[history](messages[:i])
        p, gen = _prompt_and_generated(tok, ctx, m, kw, i)
        text = p + gen
        roles = np.zeros(len(text), np.int8)
        _label_turn(text, roles, len(p), len(text), m, code_args, stats)
        out.append((text, roles))
    return out


def _prompt_and_generated(tok, ctx: list[dict], m: dict, kw: dict, i: int) -> tuple[str, str]:
    """(the prompt the server rendered, the text the model generated after it) for turn ``m`` after ``ctx``.

    The prompt is the template with its generation prompt, as llama-server sends it. The generated text is
    taken from the render of ``ctx + [m]``: what follows the history there, minus the generation prompt. The
    history render may close the model turn (``<turn|>``) where the continued turn does not (Gemma-4 after a
    tool response whose message also carries text, e.g. last-inline reasoning); only such a closure may
    differ, anything else is refused.
    """
    p = tok.apply_chat_template(ctx, add_generation_prompt=True, **kw)
    q = tok.apply_chat_template(ctx, **kw)
    a = tok.apply_chat_template([*ctx, m], **kw)
    if not p.startswith(q):
        raise NotPrefixError(f"turn {i}: the history render is not a prefix of the prompt")
    g = p[len(q):]
    for h in (q, q.removesuffix("<turn|>\n")):
        if a.startswith(h + g) and len(a) > len(h + g):
            return p, a[len(h + g):]
    raise NotPrefixError(f"turn {i}: the turn render does not continue the prompt render")


def _label_turn(text: str, roles: np.ndarray, g0: int, g1: int, m: dict, code_args, stats) -> None:
    """Roles of one generated assistant turn ``text[g0:g1]`` (teacher, code -> anchor, loop spans -> loop)."""
    while g1 > g0 and text[g1 - 1] in "\n " and not text[:g1].endswith("|>"):  # template whitespace after
        g1 -= 1  # the stop token is not generated
    roles[g0:g1] = TEACHER
    span = text[g0:g1]
    for c in m.get("tool_calls") or []:
        for key, val in ((c.get("function") or c).get("arguments") or {}).items():
            if key not in code_args or not isinstance(val, str) or not val:
                continue
            hit = _find_value(span, key, val)
            if hit is None:
                if stats is not None:
                    stats["code_args_not_found"] += 1
                continue
            roles[g0 + hit : g0 + hit + len(val)] = ANCHOR
    content = m.get("content") or ""
    base = span.rfind(content) if content else -1
    if content and base >= 0:
        for f in FENCE.finditer(content):
            roles[g0 + base + f.start(1) : g0 + base + f.end(1)] = ANCHOR
    for ls in m.get("loop_spans") or []:
        field = m.get(ls["field"]) or ""
        at = span.find(field) if field else -1
        if at < 0:
            if stats is not None:
                stats["loop_spans_not_found"] += 1
            continue
        roles[g0 + at + ls["start"] : g0 + at + ls["end"]] = LOOP


def _find_value(span: str, key: str, val: str) -> int | None:
    """Offset of a tool argument's string value in a rendered call (Gemma-4 ``key:<|"|>value<|"|>``, or JSON)."""
    for pre in (f'{key}:<|"|>', f'"{key}": "', f'"{key}":"'):
        at = span.find(pre + val)
        if at >= 0:
            return at + len(pre)
    at = span.find(val)  # other renderings: the value verbatim
    return at if at >= 0 else None


def token_roles(tok, text: str, roles: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Token ids of ``text`` and each token's role (highest-priority role among its characters)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if tok.decode(ids) != text:
        raise ValueError("tokenization does not round-trip the rendered text")
    out = np.zeros(len(ids), np.int8)
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s:
            r = roles[s:e]
            out[t] = r[PRIORITY[r].argmax()]
    return ids, out


def windows(ids: list[int], roles: np.ndarray, seq: int, stride: int, pad: int):
    """Fixed-length ``seq`` windows; each scores only tokens no earlier window scored (see module doc)."""
    n = len(ids)
    out, scored_to, e = [], 0, min(seq, n)
    while True:
        w0 = max(0, e - seq)
        r = roles[w0:e].copy()
        r[: max(0, scored_to - w0)] = IGNORE
        if (r > 0).any():
            x = list(ids[w0:e]) + [pad] * (seq - (e - w0))
            out.append((x, np.concatenate([r, np.zeros(seq - (e - w0), np.int8)])))
        scored_to = e
        if e >= n:
            return out
        e = min(e + stride, n)


def build(tok, rows, seq: int, stride: int, code_args=CODE_ARGS, history: str = "keep") -> tuple[list[dict], dict]:
    """Windows of every row plus build statistics.

    ``history``: ``keep`` = the serving client re-sent earlier reasoning (one render per segment); ``strip`` =
    it re-sent none; ``last-inline`` = only the newest, folded into the text (Cerebriline SDK >= 4.100.260 on a
    template that drops the field). ``strip`` and ``last-inline`` give one example per assistant turn.
    """
    stats = {"rows": 0, "segments": 0, "refused_not_prefix": 0, "windows": 0, "code_args_not_found": 0,
             "loop_spans_not_found": 0, "tokens_by_role": [0, 0, 0, 0]}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    for row in rows:
        stats["rows"] += 1
        msgs = _norm(row["messages"])
        if history in HISTORY_CONTEXT:
            try:
                items = render_turns(tok, msgs, row.get("tools"), history, code_args, stats)
            except NotPrefixError:
                stats["refused_not_prefix"] += 1
                continue
        else:
            items = []
            for end in segment_ends(msgs):
                try:
                    items.append(render_segment(tok, msgs, row.get("tools"), end, code_args, stats))
                except NotPrefixError:  # noqa: PERF203 (refuse one segment, keep the others)
                    stats["refused_not_prefix"] += 1
        for k, (text, roles) in enumerate(items):
            stats["segments"] += 1
            ids, troles = token_roles(tok, text, roles)
            for w, (x, r) in enumerate(windows(ids, troles, seq, stride, pad)):
                out.append({"id": f"{row.get('id', stats['rows'])}/s{k}/w{w}", "ids": x, "role": r.tolist()})
                for q in range(4):
                    stats["tokens_by_role"][q] += int((r == q).sum())
    stats["windows"] = len(out)
    return out, stats


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model dir whose tokenizer + chat template render the rows")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="output .roles.jsonl")
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--stride", type=int, default=0, help="window step (default seq // 2)")
    ap.add_argument("--code-args", default=",".join(CODE_ARGS))
    ap.add_argument("--history-reasoning", choices=["keep", *HISTORY_CONTEXT], required=True,
                    help="how the serving client re-sent earlier turns' reasoning: keep = all of it (one render per "
                    "segment); strip = none; last-inline = only the newest, as <think>..</think> in its text "
                    "(Cerebriline SDK >= 4.100.260, automaticum's opencoti provider). strip / last-inline: one "
                    "example per assistant turn")
    a = ap.parse_args()
    if not a.out.endswith(".roles.jsonl"):
        raise SystemExit("--out must end in .roles.jsonl (the trainer recognises role data by that suffix)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(ln) for ln in Path(a.inp).open() if ln.strip()]
    out, stats = build(tok, rows, a.seq, a.stride or a.seq // 2, tuple(a.code_args.split(",")), a.history_reasoning)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(w) + "\n" for w in out)
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps({**stats, "args": vars(a)}, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
