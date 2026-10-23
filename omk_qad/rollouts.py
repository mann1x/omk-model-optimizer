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

Request capture (``--capture DIR``): the Cerebriline SDK's "Save model requests" files, ``<session>/<stem>.json``
(``{at, url, method, sessionId, body}``: the exact chat-completions body sent) and ``<stem>.response.json``
(``{at, status, body, complete, error}``: the raw SSE or JSON the server returned). Each request is one
example: its messages, rendered with its own ``chat_template_kwargs`` over the server's (``--template-kwargs``),
are the context, and the response is the scored turn. No history mode is modelled; the body is what the model
was shown. A response cut short (``complete: false``, a guard's cancel, or a finish other than stop /
tool_calls) is scored up to its last generated character only: the template's closure after it was never
generated. ``--verify-server URL`` posts each body to llama-server's ``/apply-template`` and refuses any
example whose prompt differs from the builder's render.

    python -m omk_qad.rollouts --model DIR --in conv.jsonl --out X.roles.jsonl --seq 16384 [--stride 8192]
    python -m omk_qad.rollouts --model DIR --capture CAPDIR --template-kwargs '{"enable_thinking": true}' \
        [--loop-spans spans.jsonl] [--verify-server http://host:port] --out X.roles.jsonl
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

IGNORE, TEACHER, ANCHOR, LOOP = 0, 1, 2, 3
PRIORITY = np.array([0, 2, 1, 3])  # a token spanning several roles takes the highest: loop > teacher > anchor
CODE_ARGS = ("content", "new_string", "old_string", "new_str", "old_str", "code", "patch", "diff", "file_text",
             "replace", "search", "text")
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"  # Gemma-4 thinking channel
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
    r"""(the prompt the server rendered, the text the model generated after it) for turn ``m`` after ``ctx``.

    The prompt is the template with its generation prompt, as llama-server sends it. The generated text is
    taken from the render of ``ctx + [m]``: what follows the history there, minus the generation prompt. The
    history render may close the model turn (``<turn|>``) where the continued turn does not (Gemma-4 after a
    tool response whose message also carries text, e.g. last-inline reasoning); only such a closure may
    differ, anything else is refused. When the generation prompt opened the thought channel (Gemma-4 after a
    tool response) and the turn has no reasoning, the model closed it at once: the generated text starts with
    the close tag (the template's own empty-thought form is ``<|channel>thought\n<channel|>``).
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
        if g.endswith(THOUGHT_OPEN) and not m.get("reasoning_content") and a.startswith(h + g[: -len(THOUGHT_OPEN)]):
            return p, THOUGHT_CLOSE + a[len(h + g) - len(THOUGHT_OPEN):]
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


class Packer:
    """``--pack``: examples that fit one window share it back to back (each keeps its own <bos>).

    Short chat examples then do not each cost a ``seq``-long forward of padding; longer ones go through ``windows``.
    """

    def __init__(self, seq: int, pad: int):
        """An empty window of ``seq`` tokens."""
        self.seq, self.pad, self.ids, self.roles, self.names = seq, pad, [], [], []

    def add(self, name: str, ids: list[int], roles: np.ndarray) -> list[tuple[str, list[int], np.ndarray]]:
        """Add one example (``len(ids) <= seq``); returns the window it closed, if any."""
        out = self.flush() if len(self.ids) + len(ids) > self.seq else []
        self.ids += list(ids)
        self.roles.append(roles)
        self.names.append(name)
        return out

    def flush(self) -> list[tuple[str, list[int], np.ndarray]]:
        """The current window, padded (none when empty)."""
        if not self.ids:
            return []
        n = len(self.ids)
        r = np.concatenate([*self.roles, np.zeros(self.seq - n, np.int8)])
        out = [("+".join(self.names), self.ids + [self.pad] * (self.seq - n), r)]
        self.ids, self.roles, self.names = [], [], []
        return out


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


# ============================================================================================ request capture
FINISHED = ("stop", "tool_calls")
FIELD_ALIAS = {"reasoning": "reasoning_content", "text": "content"}  # automaticum export names


def parse_response(body: str) -> tuple[dict, str | None]:
    """(assistant message, finish_reason) of a chat-completions response: an SSE stream or one JSON body."""
    msg = {"role": "assistant", "reasoning_content": "", "content": ""}
    calls: dict[int, dict] = {}
    finish = None
    if body.lstrip().startswith("{"):
        ch = json.loads(body)["choices"][0]
        m = ch.get("message") or {}
        msg["reasoning_content"] = m.get("reasoning_content") or m.get("reasoning") or ""
        msg["content"] = m.get("content") or ""
        calls = dict(enumerate(m.get("tool_calls") or []))
        finish = ch.get("finish_reason")
    else:
        for ln in body.splitlines():
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            ev = json.loads(ln[5:])
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                msg["reasoning_content"] += d.get("reasoning_content") or d.get("reasoning") or ""
                msg["content"] += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", len(calls)),
                                         {"type": "function", "function": {"name": "", "arguments": ""}})
                    c["id"] = tc.get("id") or c.get("id")
                    fn = tc.get("function") or {}
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += fn.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return msg, finish


def read_capture(root: Path, loop_spans: Path | None = None):
    """Captured (id, request, response or None) in send order, with loop spans attached to their response.

    ``loop_spans``: JSONL rows ``{"request": "<session>/<stem>.json", "field", "start", "end", "source"}``
    (automaticum D106: ``source`` is ``guard`` (an SDK reasoning_loop cut) or ``detector``; ``field``
    ``reasoning`` names the response's reasoning_content).
    """
    spans: dict[str, list] = {}
    if loop_spans:
        for ln in loop_spans.open():
            if ln.strip():
                r = json.loads(ln)
                field = FIELD_ALIAS.get(r["field"], r["field"])
                spans.setdefault(r["request"], []).append(
                    {"field": field, "start": r["start"], "end": r["end"], "source": r.get("source", "detector")})
    for f in sorted(root.glob("**/*.json")):
        if f.name.endswith(".response.json"):
            continue
        rid = str(f.relative_to(root))
        rf = f.with_name(f.name[: -len(".json")] + ".response.json")
        resp = json.loads(rf.read_text()) if rf.exists() else None
        yield rid, json.loads(f.read_text()), resp, spans.get(rid, [])


def apply_template_on(url: str):
    """``verify(body) -> prompt`` through llama-server's ``/apply-template`` (the server's own render)."""
    def verify(body: dict) -> str:
        req = urllib.request.Request(url.rstrip("/") + "/apply-template", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["prompt"]
    return verify


def render_captured(tok, body: dict, resp: dict, spans: list, server_kwargs: dict, code_args=CODE_ARGS,
                    stats=None, verify=None):
    """(text, char roles) of one captured request + response, or a refusal reason (str)."""
    if resp is None:
        return "no_response"
    if resp.get("status") != 200:
        return "bad_status"
    m, finish = parse_response(resp.get("body") or "")
    cut = not resp.get("complete", False) or finish not in FINISHED
    try:
        m = _norm([m])[0]
        msgs = _norm(body["messages"])
    except json.JSONDecodeError:
        return "cut_in_tool_call" if cut else "bad_tool_arguments"
    if not (m["reasoning_content"] or m["content"] or m.get("tool_calls")):
        return "empty_response"
    if cut and m.get("tool_calls"):
        return "cut_in_tool_call"
    m["loop_spans"] = spans
    kw = {"tools": body.get("tools"), "tokenize": False, **server_kwargs, **(body.get("chat_template_kwargs") or {})}
    try:
        p, gen = _prompt_and_generated(tok, msgs, m, kw, len(msgs))
    except NotPrefixError:
        return "not_prefix"
    if verify is not None:
        served = verify({k: v for k, v in body.items() if k != "stream"})
        if served != p.removeprefix(tok.bos_token or ""):
            return "server_mismatch"
    text = p + gen
    roles = np.zeros(len(text), np.int8)
    _label_turn(text, roles, len(p), len(text), m, code_args, stats)
    if cut:
        last = (m["content"] or m["reasoning_content"]).rstrip()
        at = text.rfind(last, len(p))
        if at < 0:
            return "cut_not_located"
        roles[at + len(last):] = IGNORE  # the template's closure after a cut was never generated
    return text, roles


def build_capture(tok, items, seq: int, stride: int, server_kwargs: dict, code_args=CODE_ARGS, verify=None,
                  pack: bool = False):
    """Windows of every captured request + response (see module doc) plus build statistics."""
    stats = {"requests": 0, "examples": 0, "cut": 0, "windows": 0, "code_args_not_found": 0,
             "loop_spans_not_found": 0, "refused": {}, "tokens_by_role": [0, 0, 0, 0]}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    packer = Packer(seq, pad) if pack else None

    def emit(done):
        for name, x, r in done:
            out.append({"id": name, "ids": x, "role": r.tolist()})
            for q in range(4):
                stats["tokens_by_role"][q] += int((r == q).sum())

    for rid, req, resp, spans in items:
        stats["requests"] += 1
        got = render_captured(tok, req["body"], resp, spans, server_kwargs, code_args, stats, verify)
        if isinstance(got, str):
            stats["refused"][got] = stats["refused"].get(got, 0) + 1
            continue
        text, roles = got
        stats["examples"] += 1
        stats["cut"] += int(not resp.get("complete", False) or parse_response(resp["body"])[1] not in FINISHED)
        ids, troles = token_roles(tok, text, roles)
        if packer is not None and len(ids) <= seq:
            done = packer.add(rid, ids, troles)
        else:
            done = [(f"{rid}/w{w}", x, r) for w, (x, r) in enumerate(windows(ids, troles, seq, stride, pad))]
        emit(done)
    emit(packer.flush() if packer is not None else [])
    stats["windows"] = len(out)
    return out, stats


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model dir whose tokenizer + chat template render the rows")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="inp", help="conversations JSONL (needs --history-reasoning)")
    src.add_argument("--capture", help="SDK request-capture dir (<session>/<stem>.json + .response.json)")
    ap.add_argument("--out", required=True, help="output .roles.jsonl")
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--stride", type=int, default=0, help="window step (default seq // 2)")
    ap.add_argument("--code-args", default=",".join(CODE_ARGS))
    ap.add_argument("--history-reasoning", choices=["keep", *HISTORY_CONTEXT],
                    help="how the serving client re-sent earlier turns' reasoning: keep = all of it (one render per "
                    "segment); strip = none; last-inline = only the newest, as <think>..</think> in its text "
                    "(Cerebriline SDK >= 4.100.260, automaticum's opencoti provider). strip / last-inline: one "
                    "example per assistant turn")
    ap.add_argument("--template-kwargs", help="--capture: the server's --chat-template-kwargs JSON (request "
                    "kwargs override it), e.g. '{\"enable_thinking\": true}'; '{}' when the server sets none")
    ap.add_argument("--loop-spans", help="--capture: JSONL {request, field, start, end} (automaticum export)")
    ap.add_argument("--pack", action="store_true", help="--capture: examples shorter than --seq share windows "
                    "back to back (chat replay), instead of one padded window each")
    ap.add_argument("--verify-server", help="--capture: llama-server URL; refuse examples whose /apply-template "
                    "prompt differs from the builder's render")
    a = ap.parse_args()
    if a.inp and not a.history_reasoning:
        raise SystemExit("--in needs --history-reasoning (keep | strip | last-inline)")
    if a.capture and a.template_kwargs is None:
        raise SystemExit("--capture needs --template-kwargs: the server's own template defaults are not in the body")
    if not a.out.endswith(".roles.jsonl"):
        raise SystemExit("--out must end in .roles.jsonl (the trainer recognises role data by that suffix)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    code_args, stride = tuple(a.code_args.split(",")), a.stride or a.seq // 2
    if a.capture:
        items = read_capture(Path(a.capture), Path(a.loop_spans) if a.loop_spans else None)
        out, stats = build_capture(tok, items, a.seq, stride, json.loads(a.template_kwargs), code_args,
                                   apply_template_on(a.verify_server) if a.verify_server else None, a.pack)
    else:
        rows = [json.loads(ln) for ln in Path(a.inp).open() if ln.strip()]
        out, stats = build(tok, rows, a.seq, stride, code_args, a.history_reasoning)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(w) + "\n" for w in out)
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps({**stats, "args": vars(a)}, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
