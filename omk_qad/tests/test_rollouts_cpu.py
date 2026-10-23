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
