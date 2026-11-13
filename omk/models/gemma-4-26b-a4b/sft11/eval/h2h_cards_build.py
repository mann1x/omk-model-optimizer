#!/usr/bin/env python3
# h2h_cards_build.py REPORT_JSON RUNS_JSON ACC_BASE,ACC_SFT -- bm-889: fill the head-to-head into both HF card templates
#   (hf_cards/{weights,gguf}/README.h2h_template.md) from the FINAL h2h_report.json and write README.md.
#   The pre-h2h card is kept once as README.pre_h2h.md (never overwritten). Refuses a partial report. Uploads nothing.
import json
import os
import subprocess
import sys

H = os.path.dirname(os.path.abspath(__file__))
CARDS = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/gemma-4-26B-A4B-agentic/hf_cards"
rep_p, runs_p, acc = sys.argv[1:4]
rep = json.load(open(rep_p))
a, b = rep["arms"]["base"], rep["arms"]["sft11"]
if a["n"] != 20 or b["n"] != 20:
    sys.exit(f"REFUSE: report has base {a['n']} / sft11 {b['n']} runs, need 20/20")


def section(mode):
    return subprocess.run([sys.executable, f"{H}/h2h_card_section.py", rep_p, runs_p, mode, acc], check=True,
                          capture_output=True, text=True).stdout.strip()


ratio = rep["rates"]["base"]["reasoning_loop_cuts"] / rep["rates"]["sft11"]["reasoning_loop_cuts"]
subs = {"{SFT_PASS}": str(b["pass"]), "{BASE_PASS}": str(a["pass"]), "{SFT_FAILS}": str(b["n"] - b["pass"]),
        "{LOOP_RATIO}": f"{ratio:.0f}"}
for card, marker, mode in (("weights", "<!-- H2H_FULL -->", "full"), ("gguf", "<!-- H2H_SHORT -->", "short")):
    t = open(f"{CARDS}/{card}/README.h2h_template.md").read()
    assert t.count(marker) == 1, (card, marker)
    t = t.replace(marker, section(mode))
    for k, v in subs.items():
        t = t.replace(k, v)
    assert "{" + "SFT_" not in t and "<!-- H2H" not in t, card
    cur = f"{CARDS}/{card}/README.md"
    bak = f"{CARDS}/{card}/README.pre_h2h.md"
    if not os.path.exists(bak):
        os.rename(cur, bak)
    with open(cur + ".tmp", "w") as f:
        f.write(t)
    os.replace(cur + ".tmp", cur)
    print(f"WROTE {cur} ({len(t)} bytes)")
