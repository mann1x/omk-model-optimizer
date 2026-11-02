#!/usr/bin/env python
"""Count tool calls whose string argument swallowed a later argument, per run and gate.
Streams `automaticum logs <run> --events -json`; writes nothing but gate_swallow2.json."""
import collections, json, os, re, subprocess
os.environ.setdefault("AUTOMATICUM_CONFIG", "/mnt/sdc/automaticum/config")
REPO = "/mnt/sdc/automaticum/repo"; BIN = "../bin/automaticum"
GATES = {"floor": "k-gate-floor-manic-7fe45202-w1800-p9", "sft1": "k-gate-sft1-manic-w1800-p9",
         "s400": "k-k-gate-sft2-s400-manic-w1800-p9", "sft3": "k-gate-sft3-manic-w1800-p9",
         "sft3s400": "k-gate-sft3-s400-manic-w1800-p9", "sft4s200": "k-gate-sft4-s200-manic-w1800-p9"}
KEYS = r"(?:start_line|end_line|old_text|new_text|file_path|task_progress|occurrence)"
SW = re.compile(r"(?:^|[\s`,}\"'])" + KEYS + r"\s*:")  # bm-889 2026-11-02: also a newline/space separator (sft4 s200 run 002 intent shape)
def ids(b):
    o = subprocess.run([BIN, "run", "ls", "-batch", b], cwd=REPO, capture_output=True, text=True).stdout
    return sorted({l.split()[0] for l in o.splitlines() if l[:2] == "20"})
out = {}; 
for gate, batch in GATES.items():
    T = collections.Counter(); runs = []
    for run in ids(batch):
        p = subprocess.Popen([BIN, "logs", run, "--events", "-json"], cwd=REPO, stdout=subprocess.PIPE, text=True, errors="replace")
        c = collections.Counter(); tails = collections.Counter(); inputs = {}
        for l in p.stdout:
            if '"contentType":"tool"' not in l.replace(" ", "") : continue
            try: e = json.loads(l).get("event") or {}
            except Exception: continue
            ev = e.get("payload", {}).get("event", {}) if isinstance(e, dict) else {}
            if ev.get("contentType") != "tool": continue
            if ev.get("type") == "content_start":
                i = ev.get("input") or {}; c["calls"] += 1
                if any("<|" in k for k in i): c["bad_key"] += 1
                sw = False
                for k, v in i.items():
                    if isinstance(v, str):
                        m = SW.search(v)
                        if m: sw = True; tails[v[max(0, m.start() - 12):m.end() + 6]] += 1
                if sw: c["swallowed"] += 1
                inputs[ev.get("toolCallId")] = ev.get("toolName")
            if ev.get("type") == "content_end" and ev.get("toolName") == "editor":
                c["editor"] += 1
                o = ev.get("output"); o = o[0] if isinstance(o, list) and o else o
                err = str((o or {}).get("error") or "") if isinstance(o, dict) else ""
                if err: c["editor_err"] += 1
                if "no `start_line`" in err: c["no_start"] += 1
                if "No change" in err: c["no_change"] += 1
        p.wait(); T.update(c)
        runs.append({"run": run[-3:], **c, "top": tails.most_common(2)})
        print(gate, run[-3:], dict(c), tails.most_common(1), flush=True)
    out[gate] = {"total": T, "runs": runs}
    print("TOTAL", gate, dict(T), flush=True)
json.dump(out, open("gate_swallow2.json", "w"), indent=1, default=str)
print("SCAN_DONE", flush=True)
