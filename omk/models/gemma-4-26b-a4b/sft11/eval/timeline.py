#!/usr/bin/env python3
# timeline.py -- bm-889 loop-onset analysis (owner 2026-11-11 11:26: "find out if there's a pattern on what happens before
#   the model starts looping. is it because it can't count braces properly?"). Reads one run's
#   `automaticum logs <id> --events -json` on stdin; prints one JSON line with the run's ordered events:
#   R reasoning block: chars, maxrep (most repeats of one normalised 80-char unit, as loopscan.py), top unit, line90 quotes
#     (times the block quotes the dDec line), brace_talk (brace-count phrases)
#   T tool call: name, target line (90 dDec / 95 dBoss / 111 en.forEach / other), ok, error class, and for editor calls on
#     90/95/111: d_brace / d_paren (old minus new), same (new == old), exact90/exact95 (new == the one correct fix of the
#     ORIGINAL line), drift (old and new differ after deleting every brace and paren = changed something besides delimiters),
#     old_matches_orig (old_text equals the original planted line)
#   C checker / run_game result: error message, line
#   G runner guard trip: guard, phase
# Reads only.
import difflib
import json
import re
import sys
from collections import Counter

run = sys.argv[1]
ORIG90 = ("    dDec(c,x){this.dc.forEach(d=>{if(d.tp==='stl'){c.fillStyle='#8899aa';c.fillRect(d.x-x,d.y-20,8,20);"
          "c.fillRect(d.x-x+2,d.y-15,4,3);}else if(d.tp==='msh'){c.fillStyle='#667799';c.fillRect(d.x-x,d.y-12,10,12);"
          "c.beginPath();c.arc(d.x-x+5,d.y-12,8,Math.PI,0);c.fill();}else{c.fillStyle='#aabbcc';c.fillRect(d.x-x-4,d.y-10,"
          "8,10);c.beginPath();c.arc(d.x-x,d.y-10,5,0,6.28);c.fill();}}});}")
FIX90 = ORIG90[:-6] + "}});}"
assert ORIG90.endswith("}}});}") and FIX90.endswith("c.fill();}});}")
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
NORM = re.compile(r"[^a-z0-9]+")
BRACE_TALK = re.compile(r"\b(one|two|three|four|five|six|seven|eight|\d)\b[^.\n]{0,30}?(brace|curly|`\}+`)", re.I)


def which_line(s):
    s = s or ""
    if "dDec(" in s:
        return 90
    if "dBoss(" in s:
        return 95
    if "currentLevelObj.en.forEach(e=>{if(e.al){e.x+=e.vx" in s or "en.forEach(e=>{if(e.al){e.x+=e.vx" in s:
        return 111
    return 0


def rscore(b):
    units = [u.strip() for u in SPLIT.split(b) if len(u.strip()) >= 40]
    keys = [k for k in (NORM.sub("", u.lower())[:80] for u in units) if len(k) >= 20]
    c = Counter(keys)
    top, mr = c.most_common(1)[0] if c else ("", 0)
    return {"k": "R", "chars": len(b), "maxrep": mr, "top_line": which_line(next((u for u in units if NORM.sub("", u.lower())[:80] == top), "")) if top else 0,
            "q90": b.count("dDec("), "brace_talk": len(BRACE_TALK.findall(b))}


def strip_delims(s):
    return re.sub(r"[{}()]", "", s)


ev, cur, pend = [], [], {}
for line in sys.stdin:
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("k") == "runner" and e.get("note") == "guard_trip":
        ev.append({"k": "G", "guard": e.get("guard"), "phase": e.get("phase")})
        continue
    if e.get("k") != "event":
        continue
    x = e.get("event") or {}
    if x.get("type") != "agent_event":
        continue
    a = (x.get("payload") or {}).get("event") or {}
    t, ct = a.get("type"), a.get("contentType")
    if ct == "reasoning":
        if t == "content_start":
            cur.append(a.get("reasoning") or "")
        elif t == "content_end":
            ev.append(rscore(a.get("reasoning") or "".join(cur)))
            cur = []
        continue
    if ct != "tool":
        continue
    if t == "content_start":
        pend[a.get("toolCallId")] = a.get("input") or {}
        continue
    if t != "content_end":
        continue
    name = a.get("toolName")
    inp = pend.pop(a.get("toolCallId"), {}) or {}
    o = a.get("output")
    s = o if isinstance(o, str) else json.dumps(o)
    if name == "editor":
        old, new = inp.get("old_text") or "", inp.get("new_text") or ""
        ok = "Editor operation failed" not in s and '"success": false' not in s
        m = re.search(r"(No change|No replacement performed|This call names `end_line|Read before editing|changed since you last read|"
                      r"Invalid start_column|Invalid end_column|Missing required argument[^\"]{0,20}|cut off by the output budget)", s)
        ln = which_line(old) or which_line(new)
        r = {"k": "T", "name": name, "line": ln, "ok": ok, "err": m.group(1) if m else ("" if ok else "other")}
        if ln:
            r.update({"d_brace": old.count("}") - new.count("}"), "d_paren": old.count(")") - new.count(")"), "same": old == new,
                      "drift": strip_delims(old.strip()) != strip_delims(new.strip()), "old_matches_orig": old.strip() == ORIG90.strip() if ln == 90 else None,
                      "exact90": new.strip() == FIX90.strip() if ln == 90 else None, "old_len": len(old), "new_len": len(new)})
        if ln == 90:  # new line 90 against the ORIGINAL planted line, whatever form the call took (old_text or line range)
            # only the dDec line of new_text (multi-line edits and reformatting kept apart): its drift in characters after
            # deleting every brace and paren (0 = a pure delimiter edit), and how the call was shaped
            dl = [x for x in new.split("\n") if "dDec(" in x]
            n90 = dl[0].strip() if dl else ""
            a_, b_ = strip_delims(ORIG90.strip()), strip_delims(n90)
            sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
            r.update({"n90_lines": new.count("\n") + 1, "n90_found": bool(dl),
                      "n90_drift_chars": sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"),
                      "n90_vs": "FIX" if n90 == FIX90.strip() else "ORIG" if n90 == ORIG90.strip() else "other",
                      "n90_d_brace": ORIG90.count("}") - n90.count("}"), "n90_d_paren": ORIG90.count(")") - n90.count(")"),
                      "n90_len": len(n90)})
            ns = new.strip()
            r.update({"new_vs": "FIX" if ns == FIX90.strip() else "ORIG" if ns == ORIG90.strip() else "other",
                      "nd_brace": ORIG90.count("}") - new.count("}"), "nd_paren": ORIG90.count(")") - new.count(")"),
                      "ndrift": strip_delims(ns) != strip_delims(ORIG90.strip()), "form": "old_text" if old else "range"})
        ev.append(r)
    elif name == "check_file" or (name == "run_commands" and "run_game" in json.dumps(inp)):
        msg = re.search(r'\\?"message\\?":\\?"([^"\\]{0,60})', s)
        er = re.search(r'\\?"error\\?":\\?"([^"\\]{0,60})', s)
        lnm = re.search(r'\\?"start\\?":\{\\?"line\\?":(\d+)', s)
        ev.append({"k": "C", "name": "check" if name == "check_file" else "run_game", "err": (er.group(1) if er else "") or (msg.group(1) if msg else "clean"),
                   "line": int(lnm.group(1)) + 1 if lnm else None, "ok_game": '\\"ok\\":true' in s or '"ok":true' in s})
    else:
        tgt = json.dumps(inp)
        rl = re.search(r'"start_line": (\d+)', tgt)
        ev.append({"k": "T", "name": name, "line": int(rl.group(1)) if rl else 0, "ok": '"success": false' not in s, "err": ""})
if cur:
    ev.append(rscore("".join(cur)))
print(json.dumps({"run": run, "events": ev}))
