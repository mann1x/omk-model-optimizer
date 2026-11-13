#!/usr/bin/env python3
# run_summary.py ID VERDICT WALL_MS LOOPSCAN_JSON TIMELINE_JSON -- bm-889 sft11 live gate report (owner 2026-11-11 18:17 CET:
#   "give me results of each run as it comes"). One line per finished run from the same loopscan.py / timeline.py
#   measures as the 2026-11-11 re-score: loop blocks (one normalised 80-char unit >= 20x / >= 10x), worst repeat count,
#   guard cuts, reasoning volume, line-90 history (fixed / bug written back), the last checker or run_game result, tool calls.
# Reads only.
import json
import sys

rid, verdict, wall_ms, lp, tp = sys.argv[1:6]
L = json.loads(open(lp).read().strip().splitlines()[-1])
T = json.loads(open(tp).read().strip().splitlines()[-1])
ev = T["events"]
st90, fixed, reverted, e90, last_c, tools, guards = "ORIG", False, False, 0, "none", 0, 0
i_c = i_e = -1
for i, e in enumerate(ev):
    if e["k"] == "G":
        guards += 1
    elif e["k"] == "C":
        last_c = ("clean" if e["err"] in ("clean", "") else e["err"][:40]) + (f"@{e['line']}" if e.get("line") and e["err"] not in ("clean", "") else "")
        if e.get("ok_game"):
            last_c += " game starts"  # top-level run_game ok only; the oracle verdict is game_logic (below)
        i_c = i
        tools += 1
    elif e["k"] == "T":
        tools += 1
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

# 2026-11-12: the oracle verdict the model sees is run_game's game_logic block (ok / passed / total / failed), not the
# top-level "ok" (= the game started). Read it from the events file: the last run_game result, and whether the run went
# on to declare done (task_progress / a final text) after a failing game_logic.
orc = "no run_game"
if len(sys.argv) > 6:
    pend = {}
    for line in open(sys.argv[6]):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        a = (((x.get("event") or {}).get("payload") or {}).get("event") or {})
        if a.get("contentType") != "tool":
            continue
        if a.get("type") == "content_start":
            pend[a.get("toolCallId")] = (a.get("toolName"), json.dumps(a.get("input")))
            continue
        if a.get("type") != "content_end":
            continue
        nm, inp = pend.pop(a.get("toolCallId"), ("", ""))
        if nm != "run_commands" or "run_game" not in inp:
            continue
        try:
            res = json.loads(a.get("output")[0]["result"].split("\n")[0])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        g = res.get("game_logic")
        # v5 oracle (2026-11-12): top-level ok = runs AND game_logic.ok; "runs" carries the old "game starts" meaning.
        # Pre-v5 output has no "runs" key, where ok meant "game starts".
        if not res.get("runs", res.get("ok")):
            orc = "game does not start"
        elif g is None:
            orc = "starts, no game_logic"
        else:
            orc = (f"ORACLE OK {g.get('passed')}/{g.get('total')}" if g.get("ok") else
                   f"oracle FAIL {g.get('passed')}/{g.get('total')}: " + "; ".join(f["trap"] for f in g.get("failed", [])))
blocks = L["per_block"]
longest = max((b["chars"] for b in blocks), default=0)
print(f"{rid[-3:]} {verdict:8s} {int(wall_ms) / 60000:5.1f} min | loops: blocks>=20x {L['blocks_maxrep_ge20']}, >=10x {L['blocks_maxrep_ge10']}, "
      f"worst {L['max_maxrep']}x, guard cuts {L['guard_cut_notes']} (all guard trips {guards}) | reasoning {L['think_chars'] // 1000}k chars "
      f"in {L['blocks']} blocks, longest {longest // 1000}k | line 90: {'FIXED' if fixed else 'never fixed'}"
      f"{', bug written back' if reverted else ''}, end state {st90}, {e90} edits | last check: {last_c} ({'after' if i_c > i_e else 'before'} the last line-90 edit) | last run_game: {orc} | tool calls {tools}")
