# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the rollout -> token-role builder (omk_qad.rollouts) on the Gemma-4 tokenizer + template."""

import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from omk_qad.rollouts import (
    ANCHOR,
    IGNORE,
    LOOP,
    TEACHER,
    build,
    render_segment,
    segment_ends,
    token_roles,
    windows,
)

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
WORK = Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK)) / "recovery"
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"]}}}]
CODE = "function f(x){\n  return x+1;\n}"
CONV = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix game.js"},
    {"role": "assistant", "reasoning_content": "I should look. I should look. I should look.", "content": "",
     "loop_spans": [{"field": "reasoning_content", "start": 15, "end": 43}],
     "tool_calls": [{"id": "c1", "type": "function", "function": {
         "name": "write_file", "arguments": json.dumps({"path": "game.js", "content": CODE})}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "ok"},
    {"role": "assistant", "reasoning_content": "Done.", "content": "Here:\n```js\nx=1\n```\nFixed it."},
    {"role": "user", "content": "thanks, now add tests"},
    {"role": "assistant", "reasoning_content": "Adding tests.", "content": "Sure."},
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    from omk_qad.tiny import make_tiny

    d = WORK / "tiny_g4_s0"
    if not (d / "config.json").exists():
        make_tiny("gemma4_moe", d, seed=0)
    return AutoTokenizer.from_pretrained(d)


def _spans(tok, text, roles):
    ids, r = token_roles(tok, text, roles)
    out, cur, buf = [], None, []
    for i, q in zip(ids, r, strict=True):
        if q != cur and buf:
            out.append((cur, tok.decode(buf)))
            buf = []
        cur = q
        buf.append(i)
    out.append((cur, tok.decode(buf)))
    return out


def test_segments_split_at_later_user_messages():
    from omk_qad.rollouts import _norm

    assert segment_ends(_norm(CONV)) == [5, 7]
    assert segment_ends(_norm(CONV[:5])) == [5]


def test_text_parts_join_with_a_newline_like_the_server():
    from omk_qad.rollouts import _norm

    m = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _norm([m])[0]["content"] == "a\nb"
    img = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}]}
    assert _norm([img])[0]["content"] == img["content"]  # not text-only: left to the template


def test_sdk_editor_new_text_is_anchored(tok):
    from omk_qad.rollouts import _norm

    tools = [{"type": "function", "function": {"name": "editor", "description": "Edit a file", "parameters": {
        "type": "object", "properties": {"file_path": {"type": "string"}, "new_text": {"type": "string"}},
        "required": ["file_path", "new_text"]}}}]
    conv = [CONV[0], CONV[1], {"role": "assistant", "reasoning_content": "Edit it.", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "editor", "arguments": json.dumps({"file_path": "game.js", "new_text": CODE})}}]}]
    text, roles = render_segment(tok, _norm(conv), tools, 3)
    assert CODE in "".join(s for r, s in _spans(tok, text, roles) if r == ANCHOR)


def test_roles_of_a_gemma4_agentic_turn(tok):
    from omk_qad.rollouts import _norm

    text, roles = render_segment(tok, _norm(CONV), TOOLS, 5)
    sp = _spans(tok, text, roles)
    by = {q: "".join(s for r, s in sp if r == q) for q in (IGNORE, TEACHER, ANCHOR, LOOP)}
    assert CODE in by[ANCHOR] and "x=1" in by[ANCHOR]
    assert "<|tool_response>" in by[TEACHER] and "<tool_call|>" in by[TEACHER] and by[TEACHER].endswith("<turn|>")
    assert "response:write_file" in by[IGNORE] and "Fix game.js" in by[IGNORE]  # injected / prompt: not trained
    assert by[LOOP].strip().startswith("I should look") and "I should look." in by[TEACHER]  # first one is not loop
    assert "```" in by[TEACHER] and "```" not in by[ANCHOR]  # fences are prose, their inside is code
    assert sp[-1] == (IGNORE, "\n")  # the template newline after <turn|> is not generated
    assert any(q0 == IGNORE and s0.endswith("<tool_response|><|channel>thought\n") and q1 == TEACHER
               and s1.startswith("Done.") for (q0, s0), (q1, s1) in pairwise(sp))  # server's gen prompt: unscored


def test_earlier_segment_turns_are_context_only(tok):
    from omk_qad.rollouts import _norm

    text, roles = render_segment(tok, _norm(CONV), TOOLS, 7)
    sp = _spans(tok, text, roles)
    scored = "".join(s for r, s in sp if r != IGNORE)
    assert "Adding tests." in scored and "Fixed it" not in scored and CODE not in scored
    assert "I should look" not in text  # the template dropped turn-2 reasoning: the context the model really saw


def test_windows_score_each_token_once_with_context():
    rng = np.random.default_rng(0)
    n, seq, stride = 1000, 128, 48
    ids = list(range(n))
    roles = rng.integers(0, 4, n).astype(np.int8)
    roles[:40] = 0
    seen = np.zeros(n, int)
    for x, r in windows(ids, roles, seq, stride, pad=-1):
        assert len(x) == seq == len(r)
        w0 = x[0]
        for k in np.nonzero(r)[0]:
            t = w0 + int(k)
            seen[t] += 1
            assert roles[t] == r[k]
            assert k >= seq - stride or w0 == 0  # at least seq - stride tokens of context, or the whole prefix
    assert (seen[roles > 0] == 1).all() and (seen[roles == 0] == 0).all()


def test_build_statistics_and_refusal(tok):
    out, st = build(tok, [{"id": "a", "messages": CONV, "tools": TOOLS}], seq=64, stride=32)
    assert st["segments"] == 2 and st["refused_not_prefix"] == 0 and st["code_args_not_found"] == 0
    assert st["loop_spans_not_found"] == 0 and all(st["tokens_by_role"][q] > 0 for q in range(4))
    assert all(len(w["ids"]) == 64 == len(w["role"]) for w in out)
    bad = [dict(m) for m in CONV[:5]]
    bad[2]["loop_spans"] = [{"field": "content", "start": 0, "end": 3}]  # empty field: span cannot be placed
    _, st2 = build(tok, [{"messages": bad, "tools": TOOLS}], seq=64, stride=32)
    assert st2["loop_spans_not_found"] == 1


def test_stripped_history_one_example_per_turn(tok):
    """opencoti-provider rollouts: earlier reasoning is not re-sent, so each turn is its own example."""
    from omk_qad.rollouts import _norm, render_turns

    items = render_turns(tok, _norm(CONV), TOOLS, "strip")
    assert len(items) == 3  # one per assistant turn
    sp = [_spans(tok, t, r) for t, r in items]
    scored = ["".join(s for q, s in x if q != IGNORE) for x in sp]
    ctx = ["".join(s for q, s in x if q == IGNORE) for x in sp]
    assert "I should look" in scored[0] and "I should look" not in items[1][0]  # no reasoning re-sent in turn 2
    assert CODE in items[1][0] and CODE not in scored[1]  # turn 1's tool call is context of turn 2
    assert "Done." in scored[1] and "Done." not in items[2][0] and "Adding tests." in scored[2]
    assert all("<channel|>" not in c.split("<|turn>model")[-1] for c in ctx[1:])  # no earlier thought re-sent
    assert "<tool_response|><|channel>thought\n" in ctx[1]  # the server's generation prompt is context
    out, st = build(tok, [{"id": "a", "messages": CONV, "tools": TOOLS}], seq=256, stride=128, history="strip")
    assert st["segments"] == 3 and st["refused_not_prefix"] == 0 and len(out) == 3


def test_last_inline_history_matches_the_sdk(tok):
    """Cerebriline SDK >= 4.100.260 (ai-sdk.ts): only the newest reasoning is re-sent, as <think>R</think> text."""
    from omk_qad.rollouts import _context_last_inline, _norm, render_turns

    items = render_turns(tok, _norm(CONV), TOOLS, "last-inline")
    assert len(items) == 3
    sp = [_spans(tok, t, r) for t, r in items]
    scored = ["".join(s for q, s in x if q != IGNORE) for x in sp]
    reasoning = "I should look. I should look. I should look."
    assert f"<think>{reasoning}</think>" in items[1][0] and "<think>" not in scored[1]  # turn 1 inlined, context
    assert f"<think>{reasoning}</think>\n" not in items[1][0]  # it had no text: the block alone
    assert "I should look" not in items[2][0]  # older reasoning dropped once a newer one exists
    assert "<think>Done.</think>\nHere:" in items[2][0] and "Adding tests." in scored[2]
    assert "</think><turn|>\n<|channel>thought\n" in items[1][0]  # the server's prompt, closed turn and all
    assert "I should look" in scored[0] and "Done." in scored[1]  # each turn's own reasoning is still trained
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "reasoning_content": "only thinking"},  # emptied by the drop: omitted
            {"role": "assistant", "reasoning_content": " recap ", "content": "Plan.", "pinned_reasoning": True},
            {"role": "assistant", "reasoning_content": " newest ", "content": "ok"}]
    ctx = _context_last_inline(msgs)
    assert [m.get("content") for m in ctx] == ["go", "recap\n\nPlan.", "<think>newest</think>\nok"]
    out, st = build(tok, [{"id": "a", "messages": CONV, "tools": TOOLS}], seq=256, stride=128, history="last-inline")
    assert st["segments"] == 3 and st["refused_not_prefix"] == 0 and len(out) == 3


def _sse(*deltas, finish=None):
    ev = [{"choices": [{"delta": d, "finish_reason": None}]} for d in deltas]
    ev.append({"choices": [{"delta": {}, "finish_reason": finish}]}) if finish else None
    return "".join(f"data: {json.dumps(e)}\n\n" for e in ev) + ("data: [DONE]\n\n" if finish else "")


def _wire(messages):
    """CONV in OpenAI wire form: tool-call arguments as JSON strings, no loop_spans."""
    return [{k: v for k, v in m.items() if k != "loop_spans"} for m in messages]


def _capture(tmp_path, reqs):
    d = tmp_path / "cap" / "sess_1"
    d.mkdir(parents=True)
    for n, (body, resp) in enumerate(reqs):
        stem = d / f"2026-10-23T17-00-00-000Z-{n:04d}"
        (stem.with_suffix(".json")).write_text(json.dumps({"at": "t", "url": "u", "method": "POST", "body": body}))
        if resp is not None:
            (stem.parent / (stem.name + ".response.json")).write_text(json.dumps(resp))
    return tmp_path / "cap"


def test_request_capture(tok, tmp_path):
    """SDK capture: the body is the context, the response the scored turn; a cut turn stops at its last char."""
    from omk_qad.rollouts import _norm, build_capture, parse_response, read_capture, render_captured

    kw = {"enable_thinking": True}
    body = {"model": "g", "messages": _wire(CONV[:4]), "tools": TOOLS, "stream": True}
    done = {"status": 200, "complete": True,
            "body": _sse({"reasoning_content": "Do"}, {"reasoning_content": "ne."}, {"content": "Fixed it."},
                         finish="stop")}
    call = {"status": 200, "complete": True, "body": _sse(
        {"tool_calls": [{"index": 0, "id": "c9",
                         "function": {"name": "write_file", "arguments": '{"path": "a.js", '}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"content": "let a = 1;"}'}}]}, finish="tool_calls")}
    loop = "Hmm. Hmm. Hmm. Hmm."
    cut = {"status": 200, "complete": False, "error": "cancelled by the reader",
           "body": _sse({"reasoning_content": "Let me think. "}, {"reasoning_content": loop})}
    m, fin = parse_response(call["body"])
    assert fin == "tool_calls"
    assert m["tool_calls"][0]["function"]["arguments"] == '{"path": "a.js", "content": "let a = 1;"}'
    root = _capture(tmp_path, [(body, done), (body, call), (body, cut), (body, None)])
    rid = sorted(str(f.relative_to(root)) for f in root.glob("**/*.json") if "response" not in f.name)[2]
    spans = tmp_path / "spans.jsonl"
    spans.write_text(json.dumps({"request": rid, "field": "reasoning", "start": 14, "end": 14 + len(loop),
                                 "source": "guard"}))  # automaticum's export names (D106)
    items = list(read_capture(root, spans))
    assert [len(x[3]) for x in items] == [0, 0, 1, 0] and items[3][2] is None
    assert items[2][3] == [{"field": "reasoning_content", "start": 14, "end": 14 + len(loop), "source": "guard"}]

    text, roles = render_captured(tok, body, done, [], kw)
    sp = _spans(tok, text, roles)
    scored = "".join(s for q, s in sp if q != IGNORE)
    assert scored.startswith("Done.") and scored.endswith("Fixed it.<turn|>")  # gen prompt is context
    assert "<tool_response|><|channel>thought\n" in "".join(s for q, s in sp if q == IGNORE)
    text, roles = render_captured(tok, body, call, [], kw)
    sp = _spans(tok, text, roles)
    by = {q: "".join(s for r, s in sp if r == q) for q in (TEACHER, ANCHOR)}
    assert "let a = 1;" in by[ANCHOR] and by[TEACHER].endswith("<|tool_response>")
    assert "<tool_response|><|channel>thought\n<channel|><|tool_call>" in text  # empty thought: closed at once
    assert by[TEACHER].startswith("<channel|><|tool_call>")  # the close tag is generated, so scored
    text, roles = render_captured(tok, body, cut, items[2][3], kw)
    sp = _spans(tok, text, roles)
    assert [q for q, _ in sp if q != IGNORE] == [TEACHER, LOOP] and sp[-2][1].strip() == loop.strip()
    assert "<channel|>" in sp[-1][1] and sp[-1][0] == IGNORE  # the closure after the cut is never scored

    def ok(b):
        return tok.apply_chat_template(_norm(b["messages"]), tools=b["tools"], add_generation_prompt=True,
                                       tokenize=False, **kw).removeprefix(tok.bos_token)

    assert isinstance(render_captured(tok, body, done, [], kw, verify=ok), tuple)
    assert render_captured(tok, body, done, [], kw, verify=lambda b: "x") == "server_mismatch"
    assert render_captured(tok, body, {**done, "status": 500}, [], kw) == "bad_status"
    out, st = build_capture(tok, read_capture(root, spans), 256, 128, kw)
    assert st["requests"] == 4 and st["examples"] == 3 and st["cut"] == 1 and st["refused"] == {"no_response": 1}
    assert st["tokens_by_role"][LOOP] > 0 and st["tokens_by_role"][ANCHOR] > 0


def test_pack_shares_windows_and_keeps_roles():
    """--pack: short examples go back to back in one window, roles kept in place, the rest padding (role 0)."""
    from omk_qad.rollouts import Packer

    p = Packer(10, pad=0)
    assert p.add("a", [1, 2, 3, 4], np.array([0, 1, 1, 0], np.int8)) == []
    assert p.add("b", [5, 6, 7], np.array([0, 3, 2], np.int8)) == []
    (name, ids, r), = p.add("c", [8, 9, 10, 11], np.array([0, 1, 1, 1], np.int8))  # does not fit: closes a+b
    assert name == "a+b" and ids == [1, 2, 3, 4, 5, 6, 7, 0, 0, 0]
    assert r.tolist() == [0, 1, 1, 0, 0, 3, 2, 0, 0, 0]
    (name, ids, r), = p.flush()
    assert name == "c" and ids[:4] == [8, 9, 10, 11] and len(ids) == 10 and r[4:].tolist() == [0] * 6
    assert p.flush() == []
