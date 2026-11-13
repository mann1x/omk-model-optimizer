#!/usr/bin/env python3
# h2h_report.py -- bm-889 SDK 4.100.277 head-to-head: base gemma-4-26B-A4B-it (GPU0 :8245) vs sft11 final (GPU1 :8244),
#   gate-owner/manic_miner_plain, 20 runs each, identical arm except model/endpoint (owner 2026-11-13 00:42 CET: "make a
#   proper comparison, not only pass rate ... loopings, length, wall time, guards, behavioral, everything that could matter
#   for a 1:1 comparison").
#
# Runs ON bs2. Reads only: automaticum `run show -json` (cached once per run as <live>/<id>.show.json, never rewritten),
#   the per-run loopscan.json / timeline.json / events.jsonl the live watcher already saved (sft11_run_summary.sh).
# Writes <out>/h2h_runs.json (one row per run), <out>/h2h_report.json (per-arm aggregates + tests), <out>/h2h_report.md.
# Pure stdlib (bs2 has no scipy): Fisher exact, Wilson CI, Mann-Whitney U (normal approx, tie-corrected).
#
# usage: h2h_report.py OUT_DIR  ARM_LABEL:ARM_NAME:BATCH_PREFIX:LIVE_DIR  ARM_LABEL:...   [--partial]
import json
import math
import os
import statistics as st
import subprocess
import sys

REPO = "/mnt/sdc/automaticum/repo"
ENV = dict(os.environ, AUTOMATICUM_CONFIG="/mnt/sdc/automaticum/config")


def sh(args):
    return subprocess.run(args, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=600).stdout


def run_ids(arm, prefix):
    rows = json.loads(sh(["../bin/automaticum", "run", "ls", "-json", "-limit", "200", "-arm-name", arm]) or "[]")
    return sorted(r["id"] for r in rows if r["id"].startswith(prefix) and r.get("verdict")), len(
        [r for r in rows if r["id"].startswith(prefix)])


def show(rid, live):
    p = f"{live}/{rid}.show.json"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        out = sh(["../bin/automaticum", "run", "show", "-json", rid])
        json.loads(out)  # refuse to cache garbage
        with open(p + ".tmp", "w") as f:
            f.write(out)
        os.replace(p + ".tmp", p)
    return json.load(open(p))


def last_json(path):
    return json.loads(open(path).read().strip().splitlines()[-1])


def line90(timeline):
    """Same rules as sft11_run_summary.py: editor edits that touch line 90 and the state they leave it in."""
    st90, fixed, reverted, e90 = "ORIG", False, False, 0
    i_c = i_e = -1
    for i, e in enumerate(timeline["events"]):
        if e["k"] == "C":
            i_c = i
        elif e["k"] == "T":
            if e["name"] == "restore_file":
                st90 = "ORIG"
            if e["name"] == "editor" and e["line"] == 90:
                e90 += 1
                i_e = i
                if e["ok"]:
                    st90 = e.get("new_vs", "other")
                    if st90 == "FIX":
                        fixed = True
                    elif st90 == "ORIG" and fixed:
                        reverted = True
    return {"fixed": fixed, "bug_written_back": reverted, "end_state": st90, "edits_line90": e90,
            "checked_after_last_line90_edit": i_c > i_e}


def scan_events(path):
    """One pass over the events stream: run_game results in order (time, game starts?, oracle passed/total), the first
    successful oracle, ask_lsp / check_file use, reasoning-loop cut notices, tool-call parse recoveries, final text."""
    t0 = None
    games, pend = [], {}
    first_ok_s = None
    final_text = ""
    done_reason = None
    max_calls_one_turn = 0  # 2026-11-13: sft11 011 emitted 310 identical editor calls in ONE response (degenerate burst)
    for line in open(path):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        ev = x.get("event") or {}
        if ev.get("type") == "status" and t0 is None:
            t0 = x["t"]
        if ev.get("type") != "agent_event":
            continue
        a = ev["payload"]["event"]
        ty = a.get("type")
        if ty == "done":
            done_reason = a.get("reason")
        if ty == "iteration_end":
            max_calls_one_turn = max(max_calls_one_turn, a.get("toolCallCount") or 0)
        if ty == "content_end" and a.get("contentType") == "text":
            final_text = a.get("text") or ""
        if a.get("contentType") != "tool":
            continue
        if ty == "content_start":
            pend[a.get("toolCallId")] = (a.get("toolName"), json.dumps(a.get("input")))
            continue
        if ty != "content_end":
            continue
        nm, inp = pend.pop(a.get("toolCallId"), ("", ""))
        if nm != "run_commands" or "run_game" not in inp:
            continue
        try:
            res = json.loads(a.get("output")[0]["result"].split("\n")[0])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        g = res.get("game_logic") or {}
        starts = bool(res.get("runs", res.get("ok")))
        el = ts_s(x["t"]) - ts_s(t0) if t0 else None
        games.append({"t_s": el, "starts": starts, "passed": g.get("passed"), "total": g.get("total"), "ok": bool(g.get("ok"))})
        if g.get("ok") and first_ok_s is None:
            first_ok_s = el
    return {"run_game": games, "first_oracle_ok_s": first_ok_s, "final_text_chars": len(final_text), "done_reason": done_reason,
            "max_calls_one_turn": max_calls_one_turn}


def ts_s(t):
    from datetime import datetime
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def collect(label, arm, prefix, live):
    ids, queued_total = run_ids(arm, prefix)
    rows = []
    for rid in ids:
        S = show(rid, live)
        M = {m["name"]: m["value"] for m in S["metrics"]}
        L = last_json(f"{live}/{rid}.loopscan.json")
        T = last_json(f"{live}/{rid}.timeline.json")
        E = scan_events(f"{live}/{rid}.events.jsonl")
        F = {}
        for f in S.get("findings") or []:
            F[f["detector"]] = F.get(f["detector"], 0) + (f.get("count") or 1)
        blocks = L["per_block"]
        last_game = E["run_game"][-1] if E["run_game"] else None
        if S["verdict"] == "PASS":
            fail_mode = "pass"
        elif last_game is None:
            fail_mode = "never ran the game"
        elif not last_game["starts"]:
            fail_mode = "game does not start"
        else:
            fail_mode = f"starts, oracle {last_game['passed']}/{last_game['total']}"
        rows.append({
            "arm": label, "id": rid, "verdict": S["verdict"], "wall_s": S["wall_ms"] / 1000, "agent_end": S.get("agent_end"),
            "claimed_success": S.get("claimed_success"), "fail_mode": fail_mode, "model": S["model"],
            "sampler": S["resolved_arm"].get("sampler"), "sdk": S["sdk_release"],
            "loops": {"blocks": L["blocks"], "think_chars": L["think_chars"], "max_repeat": L["max_maxrep"],
                      "blocks_rep_ge10": L["blocks_maxrep_ge10"], "blocks_rep_ge20": L["blocks_maxrep_ge20"],
                      "chars_in_rep_ge10": L.get("chars_in_maxrep_ge10", 0), "reasoning_loop_cuts": L["guard_cut_notes"],
                      "longest_block_chars": max((b["chars"] for b in blocks), default=0)},
            "line90": line90(T), "events": E, "findings": F, "metrics": M,
        })
    return rows, queued_total


# ---------- statistics ----------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]], two-sided by summing tables with P <= P(observed)."""
    n1, n2, k = a + b, c + d, a + c
    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    po = p(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= po * (1 + 1e-9)))


def mann_whitney(x, y):
    """Two-sided MWU, normal approximation with tie correction and continuity correction. Returns (U_x, p)."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (None, None)
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, i, ties = [0.0] * len(allv), 0, 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return (u1, 1.0)
    zz = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return (u1, math.erfc(max(zz, 0) / math.sqrt(2)))


def q(v):
    v = sorted(v)
    if not v:
        return None
    def pct(p):
        k = (len(v) - 1) * p
        f = math.floor(k)
        return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)
    return {"n": len(v), "median": pct(0.5), "p25": pct(0.25), "p75": pct(0.75), "mean": st.fmean(v), "min": v[0], "max": v[-1],
            "sum": sum(v)}


# ---------- per-run feature table ----------
FEATURES = [
    # (key, label, extractor, group)
    ("wall_min", "wall time per run (min)", lambda r: r["wall_s"] / 60, "time"),
    ("model_min", "model time (min)", lambda r: r["metrics"].get("time.model_ms", 0) / 60000, "time"),
    ("tools_min", "tool time (min)", lambda r: r["metrics"].get("time.tools_ms", 0) / 60000, "time"),
    ("turn_median_s", "median turn (s)", lambda r: r["metrics"].get("time.turn_median_ms", 0) / 1000, "time"),
    ("turn_max_s", "longest turn (s)", lambda r: r["metrics"].get("time.turn_max_ms", 0) / 1000, "time"),
    ("first_oracle_ok_min", "time to first passing oracle (min, passes only)",
     lambda r: (r["events"]["first_oracle_ok_s"] or 0) / 60 if r["events"]["first_oracle_ok_s"] else None, "time"),
    ("reasoning_loop_cuts", "reasoning-loop guard cuts", lambda r: r["loops"]["reasoning_loop_cuts"], "loops"),
    ("guard_trips", "guard trips, all kinds", lambda r: r["metrics"].get("guard_trips", 0), "loops"),
    ("loop_det_soft", "repeated-tool-call warnings (soft)", lambda r: r["metrics"].get("guard_trips.loop_detection.soft", 0), "loops"),
    ("loop_det_hard", "repeated-tool-call last warnings (hard)", lambda r: r["metrics"].get("guard_trips.loop_detection.hard", 0), "loops"),
    ("blocks_rep_ge10", "reasoning blocks with a unit repeated >=10x", lambda r: r["loops"]["blocks_rep_ge10"], "loops"),
    ("blocks_rep_ge20", "reasoning blocks with a unit repeated >=20x", lambda r: r["loops"]["blocks_rep_ge20"], "loops"),
    ("max_repeat", "worst repeat count in any block", lambda r: r["loops"]["max_repeat"], "loops"),
    ("chars_in_rep_ge10_k", "reasoning chars inside >=10x blocks (k)", lambda r: r["loops"]["chars_in_rep_ge10"] / 1000, "loops"),
    ("edit_oscillation", "edit-oscillation detector firings", lambda r: r["findings"].get("edit_oscillation", 0), "loops"),
    ("failed_identical_call", "identical failed call re-sent", lambda r: r["findings"].get("failed_identical_call", 0), "loops"),
    ("nudge_no_tool_call", "turns ended without a call (nudged)", lambda r: r["metrics"].get("guard_trips.nudge.no_tool_call", 0), "loops"),
    ("think_chars_k", "reasoning chars (k)", lambda r: r["loops"]["think_chars"] / 1000, "length"),
    ("think_blocks", "reasoning blocks", lambda r: r["loops"]["blocks"], "length"),
    ("longest_block_k", "longest reasoning block (k chars)", lambda r: r["loops"]["longest_block_chars"] / 1000, "length"),
    ("think_block_p50", "median reasoning block (chars)", lambda r: r["metrics"].get("thinking.chars.p50", 0), "length"),
    ("tok_out_k", "generated tokens (k)", lambda r: r["metrics"].get("tokens.completion", 0) / 1000, "length"),
    ("tok_reasoning_k", "reasoning tokens (k)", lambda r: r["metrics"].get("tokens.completion.reasoning", 0) / 1000, "length"),
    ("reasoning_share", "reasoning share of generated tokens",
     lambda r: (r["metrics"].get("tokens.completion.reasoning", 0) / r["metrics"]["tokens.completion"]) if r["metrics"].get("tokens.completion") else None, "length"),
    # 2026-11-13: base 001 put 93k of its 156k generated tokens into tool-call INPUTS (whole-file editor rewrites, one cut by
    # the output cap) -- that, not missing reasoning, is why its reasoning share is low. Measured separately.
    ("tok_tool_input_k", "tokens written into tool-call inputs (k)", lambda r: r["metrics"].get("tokens.completion.tool_input", 0) / 1000, "length"),
    ("tool_input_tok_per_call", "tool-input tokens per tool call",
     lambda r: r["metrics"].get("tokens.completion.tool_input", 0) / r["metrics"]["tool_calls"] if r["metrics"].get("tool_calls") else None, "length"),
    ("max_calls_one_turn", "most tool calls in one response", lambda r: r["events"]["max_calls_one_turn"], "loops"),
    # 2026-11-13: both sft11 bursts (011: 310, 019: 320 calls in one response) were editor calls whose start_line the
    # tool swallowed -> refused every time -> degenerate re-send. automaticum's swallowed_argument detector counts them.
    ("swallowed_argument", "tool calls with a swallowed argument", lambda r: r["findings"].get("swallowed_argument", 0), "behaviour"),
    ("requests_failed", "model requests cut or failed", lambda r: r["metrics"].get("requests.failed", 0), "loops"),
    ("tok_prompt_k", "prompt tokens processed (k)", lambda r: r["metrics"].get("tokens.prompt", 0) / 1000, "length"),
    ("ctx_peak_k", "peak context (k tokens)", lambda r: r["metrics"].get("context.peak_tokens", 0) / 1000, "length"),
    ("compactions", "context compactions", lambda r: r["metrics"].get("context.compactions", 0), "length"),
    ("final_text_chars", "final answer length (chars)", lambda r: r["events"]["final_text_chars"], "length"),
    ("turns", "turns (model requests)", lambda r: r["metrics"].get("turns", 0), "behaviour"),
    ("tool_calls", "tool calls", lambda r: r["metrics"].get("tool_calls", 0), "behaviour"),
    ("tool_calls_per_model_min", "tool calls per model-minute",
     lambda r: r["metrics"].get("tool_calls", 0) / (r["metrics"]["time.model_ms"] / 60000) if r["metrics"].get("time.model_ms") else None, "behaviour"),
    ("tok_per_tool_call", "generated tokens per tool call",
     lambda r: r["metrics"].get("tokens.completion", 0) / r["metrics"]["tool_calls"] if r["metrics"].get("tool_calls") else None, "behaviour"),
    ("tool_calls_failed", "failed tool calls", lambda r: r["metrics"].get("tool_calls.failed", 0), "behaviour"),
    ("edits", "edits", lambda r: r["metrics"].get("edits", 0), "behaviour"),
    ("edits_failed", "edits refused / failed", lambda r: r["metrics"].get("edits.failed", 0), "behaviour"),
    ("edits_without_check", "edits not followed by a check", lambda r: r["metrics"].get("edits.without_check", 0), "behaviour"),
    ("restores", "restore_file (start over)", lambda r: r["metrics"].get("edits.restores", 0), "behaviour"),
    ("run_game_calls", "run_game calls", lambda r: len(r["events"]["run_game"]), "behaviour"),
    ("reads", "read_files calls", lambda r: r["metrics"].get("tool_calls.by_tool.read_files", 0), "behaviour"),
    ("greps", "grep calls", lambda r: r["metrics"].get("tool_calls.by_tool.grep", 0), "behaviour"),
    ("check_file", "check_file calls (LSP)", lambda r: r["metrics"].get("tool_calls.by_tool.check_file", 0), "lsp"),
    ("ask_lsp", "ask_lsp calls (LSP)", lambda r: r["metrics"].get("tool_calls.by_tool.ask_lsp", 0), "lsp"),
    ("lsp_edits_checked", "edits auto-checked by the LSP", lambda r: r["metrics"].get("lsp.edits_checked", 0), "lsp"),
    ("lsp_edit_reports", "edits that got an LSP diagnostics report", lambda r: r["metrics"].get("lsp.edit_reports", 0), "lsp"),
    ("lsp_edits_regressed", "edits the LSP flagged as regressions", lambda r: r["metrics"].get("lsp.edits_regressed", 0), "lsp"),
    ("gen_tok_s", "server generation speed (tok/s)", lambda r: r["metrics"].get("rate.generation_tok_s"), "serving"),
    ("prompt_tok_s", "server prompt speed (tok/s)", lambda r: r["metrics"].get("rate.prompt_tok_s"), "serving"),
    ("first_token_ms", "time to first token (ms)", lambda r: r["metrics"].get("time.first_token_ms"), "serving"),
    ("parse_recoveries", "tool-call parse recoveries", lambda r: r["metrics"].get("notices.status.tool_call_parse_recovery.started", 0), "hygiene"),
    ("control_markers", "leaked control markers", lambda r: r["metrics"].get("hygiene.control_markers", 0), "hygiene"),
    ("unclosed_thinking", "unclosed thinking blocks", lambda r: r["metrics"].get("hygiene.unclosed_thinking", 0), "hygiene"),
    ("cap_hits", "output-cap hits", lambda r: r["metrics"].get("cap.hits", 0), "hygiene"),
    ("empty_completions", "empty completions", lambda r: r["metrics"].get("completions.empty", 0), "hygiene"),
]

RATE_KEYS = [("reasoning_loop_cuts", "reasoning-loop cuts"), ("guard_trips", "guard trips (all)"),
             ("loop_det_soft", "repeated-call warnings"), ("loop_det_hard", "repeated-call last warnings"),
             ("blocks_rep_ge10", "blocks >=10x"), ("edit_oscillation", "edit oscillations")]


def aggregate(rows_by_arm, labels):
    out = {"arms": {}, "tests": {}, "features": {}, "rates": {}, "time_to_pass": {}, "fail_modes": {}, "tool_profile": {},
           "failed_call_classes": {}, "guard_kinds": {}, "parity": {}}
    for lab in labels:
        R = rows_by_arm[lab]
        n, k = len(R), sum(r["verdict"] == "PASS" for r in R)
        out["arms"][lab] = {
            "n": n, "pass": k, "pass_rate": k / n if n else None, "wilson95": wilson(k, n),
            "verdicts": {v: sum(r["verdict"] == v for r in R) for v in sorted({r["verdict"] for r in R})},
            "claimed_success_but_failed": sum(bool(r["claimed_success"]) and r["verdict"] != "PASS" for r in R),
            "declared_done_and_passed": sum(r["events"]["done_reason"] == "completed" and r["verdict"] == "PASS" for r in R),
            "line90_fixed": sum(r["line90"]["fixed"] for r in R), "line90_bug_written_back": sum(r["line90"]["bug_written_back"] for r in R),
            "runs_without_reasoning_loop_cut": sum(r["loops"]["reasoning_loop_cuts"] == 0 for r in R),
            "runs_with_hard_loop_warning": sum(r["metrics"].get("guard_trips.loop_detection.hard", 0) > 0 for r in R),
            "runs_stopped_by_harness_for_looping": sum(r["metrics"].get("guard_trips.loop_detection.stopped", 0) > 0 for r in R),
            "runs_with_burst_ge10_calls": sum(r["events"]["max_calls_one_turn"] >= 10 for r in R),
            "runs_without_any_guard_trip": sum(r["metrics"].get("guard_trips", 0) == 0 for r in R),
            "best_partial_oracle_on_fails": sorted([f"{g['passed']}/{g['total']}" for r in R if r["verdict"] != "PASS"
                                                    for g in [max((g for g in r["events"]["run_game"] if g["passed"] is not None),
                                                                  key=lambda g: g["passed"], default=None)] if g]),
            "model_hours": sum(r["metrics"].get("time.model_ms", 0) for r in R) / 3.6e6,
            "wall_hours": sum(r["wall_s"] for r in R) / 3600,
        }
        fm = {}
        for r in R:
            fm[r["fail_mode"]] = fm.get(r["fail_mode"], 0) + 1
        out["fail_modes"][lab] = fm
        prof, fcls, gk = {}, {}, {}
        for r in R:
            for key, v in r["metrics"].items():
                if key.startswith("tool_calls.by_tool."):
                    prof[key[19:]] = prof.get(key[19:], 0) + v
                elif key.startswith("tool_calls.failed.by_class."):
                    fcls[key[27:]] = fcls.get(key[27:], 0) + v
                elif key.startswith("guard_trips.") and not key.startswith("guard_trips.next."):
                    gk[key[12:]] = gk.get(key[12:], 0) + v
        out["tool_profile"][lab], out["failed_call_classes"][lab], out["guard_kinds"][lab] = prof, fcls, gk
        tot = sum(prof.values()) or 1
        out["tool_profile"][lab + "_share"] = {t: v / tot for t, v in prof.items()}
        # pass curve: share of ALL runs that had passed by t minutes (wall clock of the run)
        out["time_to_pass"][lab] = {t: sum(r["verdict"] == "PASS" and r["wall_s"] <= t * 60 for r in R) / n if n else None
                                    for t in (5, 10, 15, 20, 25, 30, 31)}
        mh = out["arms"][lab]["model_hours"]
        out["rates"][lab] = {key: (sum(f(r) or 0 for r in R) / mh if mh else None) for key, _ in RATE_KEYS
                             for f in [dict((k2, f2) for k2, _, f2, _ in FEATURES)[key]]}
        samplers = {json.dumps(r["sampler"], sort_keys=True) for r in R}
        out["parity"][lab] = {"samplers": sorted(samplers), "sdk": sorted({r["sdk"] for r in R}), "models": sorted({r["model"] for r in R})}

    a, b = labels
    A, B = out["arms"][a], out["arms"][b]
    out["tests"]["pass_fisher_two_sided_p"] = fisher_two_sided(A["pass"], A["n"] - A["pass"], B["pass"], B["n"] - B["pass"])
    for key, label, f, grp in FEATURES:
        xa = [v for v in (f(r) for r in rows_by_arm[a]) if v is not None]
        xb = [v for v in (f(r) for r in rows_by_arm[b]) if v is not None]
        u, p = mann_whitney(xa, xb)
        out["features"][key] = {"label": label, "group": grp, a: q(xa), b: q(xb), "mwu_p": p}
        # same feature restricted to failed runs and to passed runs (separates "passes faster" from "behaves differently")
        for sub, cond in (("fails", lambda r: r["verdict"] != "PASS"), ("passes", lambda r: r["verdict"] == "PASS")):
            sa = [v for v in (f(r) for r in rows_by_arm[a] if cond(r)) if v is not None]
            sb = [v for v in (f(r) for r in rows_by_arm[b] if cond(r)) if v is not None]
            out["features"][key][sub] = {a: q(sa), b: q(sb), "mwu_p": mann_whitney(sa, sb)[1]}
    return out


def fmt(v, nd=1):
    if v is None:
        return "–"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def md(rep, rows_by_arm, labels, partial, queued):
    a, b = labels
    A, B = rep["arms"][a], rep["arms"][b]
    L = []
    L.append(f"# Head-to-head, SDK 4.100.277: {a} vs {b}\n")
    if partial:
        L.append(f"**PARTIAL**: {A['n']}/{queued[a]} {a} runs and {B['n']}/{queued[b]} {b} runs finished.\n")
    L.append("## Outcome\n")
    L.append(f"| | {a} | {b} |\n|---|---|---|")
    L.append(f"| pass | **{A['pass']} / {A['n']}** ({100 * A['pass_rate']:.0f} %) | **{B['pass']} / {B['n']}** ({100 * B['pass_rate']:.0f} %) |")
    L.append(f"| 95 % Wilson interval | {100 * A['wilson95'][0]:.0f}–{100 * A['wilson95'][1]:.0f} % | {100 * B['wilson95'][0]:.0f}–{100 * B['wilson95'][1]:.0f} % |")
    L.append(f"| verdicts | {A['verdicts']} | {B['verdicts']} |")
    L.append(f"| claimed success but failed | {A['claimed_success_but_failed']} | {B['claimed_success_but_failed']} |")
    # 2026-11-13: line-90 rows NOT shown -- the tracker follows line numbers, and both h2h passes restore_file'd and then
    # fixed the file with line-less search/replace edits it cannot see (passes ended "ORIG"). Kept in the JSON only.
    L.append(f"| runs that used restore_file (start over from the original) | {sum(r['metrics'].get('edits.restores', 0) > 0 for r in rows_by_arm[a])} | "
             f"{sum(r['metrics'].get('edits.restores', 0) > 0 for r in rows_by_arm[b])} |")
    L.append(f"| runs with no reasoning-loop cut | {A['runs_without_reasoning_loop_cut']} | {B['runs_without_reasoning_loop_cut']} |")
    L.append(f"| runs with no guard trip at all | {A['runs_without_any_guard_trip']} | {B['runs_without_any_guard_trip']} |")
    L.append(f"| runs that reached a hard repeated-call warning | {A['runs_with_hard_loop_warning']} | {B['runs_with_hard_loop_warning']} |")
    L.append(f"| runs stopped by the harness for looping | {A['runs_stopped_by_harness_for_looping']} | {B['runs_stopped_by_harness_for_looping']} |")
    L.append(f"| runs with a burst of >=10 tool calls in one response | {A['runs_with_burst_ge10_calls']} | {B['runs_with_burst_ge10_calls']} |")
    L.append(f"| model-hours / wall-hours spent | {A['model_hours']:.2f} / {A['wall_hours']:.2f} | {B['model_hours']:.2f} / {B['wall_hours']:.2f} |")
    L.append(f"\nFisher exact (two-sided) on pass/fail: p = {rep['tests']['pass_fisher_two_sided_p']:.4f}\n")
    L.append("### Failure modes (last run_game of each run)\n")
    modes = sorted(set(rep["fail_modes"][a]) | set(rep["fail_modes"][b]))
    L.append(f"| mode | {a} | {b} |\n|---|---|---|")
    for m in modes:
        L.append(f"| {m} | {rep['fail_modes'][a].get(m, 0)} | {rep['fail_modes'][b].get(m, 0)} |")
    L.append(f"\nBest partial oracle score reached in failed runs: {a} {A['best_partial_oracle_on_fails'] or '–'}; "
             f"{b} {B['best_partial_oracle_on_fails'] or '–'}\n")
    L.append("### Pass curve (share of all runs passed by minute t)\n")
    ts = [5, 10, 15, 20, 25, 30]
    L.append("| | " + " | ".join(f"{t} min" for t in ts) + " |\n|---|" + "---|" * len(ts))
    for lab in labels:
        L.append(f"| {lab} | " + " | ".join(f"{100 * rep['time_to_pass'][lab][t]:.0f} %" for t in ts) + " |")
    L.append("\n### Loop rates per model-hour (sum over runs / sum of model time)\n")
    L.append(f"| | {a} | {b} |\n|---|---|---|")
    for key, lab in RATE_KEYS:
        L.append(f"| {lab} | {fmt(rep['rates'][a][key])} | {fmt(rep['rates'][b][key])} |")
    groups = [("time", "Time"), ("loops", "Loops and guards (per run)"), ("length", "Length"), ("behaviour", "Behaviour"),
              ("lsp", "LSP"), ("hygiene", "Output hygiene"), ("serving", "Serving (parity check: same GPU model, same server)")]
    for g, title in groups:
        L.append(f"\n### {title}\n")
        L.append("Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.\n")
        L.append(f"| metric | {a} | {b} | p | {a} fails | {b} fails | p (fails) |\n|---|---|---|---|---|---|---|")
        for key, F in rep["features"].items():
            if F["group"] != g:
                continue
            def cell(s):
                return "–" if not s else f"{fmt(s['median'])} [{fmt(s['p25'])}–{fmt(s['p75'])}]"
            fa, fb = F["fails"][a], F["fails"][b]
            L.append(f"| {F['label']} | {cell(F[a])} | {cell(F[b])} | {fmt(F['mwu_p'], 3) if F['mwu_p'] is not None else '–'} | "
                     f"{'–' if not fa else fmt(fa['median'])} | {'–' if not fb else fmt(fb['median'])} | "
                     f"{fmt(F['fails']['mwu_p'], 3) if F['fails']['mwu_p'] is not None else '–'} |")
    L.append("\n### Tool-use profile (all runs, share of calls)\n")
    tools = sorted(set(rep["tool_profile"][a]) | set(rep["tool_profile"][b]), key=lambda t: -(rep["tool_profile"][a].get(t, 0) + rep["tool_profile"][b].get(t, 0)))
    L.append(f"| tool | {a} calls | {a} share | {b} calls | {b} share |\n|---|---|---|---|---|")
    for t in tools:
        L.append(f"| {t} | {rep['tool_profile'][a].get(t, 0)} | {100 * rep['tool_profile'][a + '_share'].get(t, 0):.0f} % | "
                 f"{rep['tool_profile'][b].get(t, 0)} | {100 * rep['tool_profile'][b + '_share'].get(t, 0):.0f} % |")
    L.append("\n### Guard trips by kind (sum)\n")
    gks = sorted(set(rep["guard_kinds"][a]) | set(rep["guard_kinds"][b]))
    L.append(f"| guard | {a} | {b} |\n|---|---|---|")
    for g in gks:
        L.append(f"| {g} | {rep['guard_kinds'][a].get(g, 0)} | {rep['guard_kinds'][b].get(g, 0)} |")
    L.append("\n### Failed tool calls by class (sum)\n")
    fcs = sorted(set(rep["failed_call_classes"][a]) | set(rep["failed_call_classes"][b]))
    L.append(f"| class | {a} | {b} |\n|---|---|---|")
    for c in fcs:
        L.append(f"| {c[:90]} | {rep['failed_call_classes'][a].get(c, 0)} | {rep['failed_call_classes'][b].get(c, 0)} |")
    L.append("\n### Per run\n")
    L.append("| arm | run | verdict | min | cuts | trips | ≥10x | worst | think k | tok k | calls | edits | run_game | end state |\n|" + "---|" * 14)
    for lab in labels:
        for r in rows_by_arm[lab]:
            L.append(f"| {lab} | {r['id'][-3:]} | {r['verdict']} | {r['wall_s'] / 60:.1f} | {r['loops']['reasoning_loop_cuts']} | "
                     f"{r['metrics'].get('guard_trips', 0)} | {r['loops']['blocks_rep_ge10']} | {r['loops']['max_repeat']} | "
                     f"{r['loops']['think_chars'] // 1000} | {r['metrics'].get('tokens.completion', 0) // 1000} | {r['metrics'].get('tool_calls', 0)} | "
                     f"{r['metrics'].get('edits', 0)} | {len(r['events']['run_game'])} | {r['fail_mode']} |")
    L.append("\n### Basis parity\n")
    for lab in labels:
        L.append(f"- {lab}: models {rep['parity'][lab]['models']}, SDK {rep['parity'][lab]['sdk']}, samplers {rep['parity'][lab]['samplers']}")
    return "\n".join(L) + "\n"


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    partial = "--partial" in sys.argv
    out = args[0]
    os.makedirs(out, exist_ok=True)
    labels, rows_by_arm, queued = [], {}, {}
    for spec in args[1:]:
        lab, arm, prefix, live = spec.split(":")
        labels.append(lab)
        rows_by_arm[lab], queued[lab] = collect(lab, arm, prefix, live)
    if not partial:
        for lab in labels:
            if len(rows_by_arm[lab]) != queued[lab]:
                sys.exit(f"REFUSE: {lab} has {len(rows_by_arm[lab])}/{queued[lab]} finished runs (pass --partial for an interim report)")
    rep = aggregate(rows_by_arm, labels)
    slim = [{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]} for lab in labels for r in rows_by_arm[lab]]
    sfx = "_partial" if partial else ""
    for name, obj in ((f"h2h_runs{sfx}.json", slim), (f"h2h_report{sfx}.json", rep)):
        with open(f"{out}/{name}.tmp", "w") as f:
            json.dump(obj, f, indent=1, default=str)
        os.replace(f"{out}/{name}.tmp", f"{out}/{name}")
    text = md(rep, rows_by_arm, labels, partial, queued)
    with open(f"{out}/h2h_report{sfx}.md.tmp", "w") as f:
        f.write(text)
    os.replace(f"{out}/h2h_report{sfx}.md.tmp", f"{out}/h2h_report{sfx}.md")
    print(f"WROTE {out}/h2h_report{sfx}.md  ({', '.join(f'{lab} {len(rows_by_arm[lab])}/{queued[lab]}' for lab in labels)})")


if __name__ == "__main__":
    main()
