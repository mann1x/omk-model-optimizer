#!/usr/bin/env python3
"""bm-893 sft12-98e corpus (owner 2026-11-14 20:27 CET: "go, build the sft12 corpus").

Selects segments from sft11's rendered train windows (sft11_build/render, --history-reasoning strip --pack, one
assistant turn scored per segment) and re-packs them with the renderer's own Packer:
  * families kept: agentic, opencode, hermes, kimi31b (the 98e agentic map's job; short turns) + an ifchat anchor
    (ifchat_v1 + ifchat_v2) for IFEval; dropped: multilingual, science, cp, math, swe (long turns, capacity the
    prune removed: 98e-init vs 128e-sft11 nll +0.15..+0.41 on the long chat families, ~0 on code).
  * no long traces: a segment whose scored (role 1) tokens exceed --max-scored is dropped, not truncated (a cut
    turn teaches a turn that never ends).
  * ifchat anchor sized to --ifchat-share of the estimated loss (held-out init nll x scored tokens), seeded subsample.
Segments are found by <bos> (each packed example keeps its own) and checked against the window id's '+' names.
Writes OUT/<fam>.train.roles.jsonl + .stats.json ("complete": true, windows, tokens_by_role, provenance) and
OUT/selection.json. Eval files are hard-linked unchanged so nll stays comparable with sft11 / the mlk8 run.
Usage: sft12_select.py SRC_RENDER OUT [--max-scored 1024] [--ifchat-share 0.10] [--seed 0]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft2_code")
from omk_qad.rollouts import N_ROLES, Packer  # noqa: E402

KEEP = ["agentic", "opencode", "hermes", "kimi31b"]
ANCHOR = ["ifchat_v1", "ifchat_v2"]
DROP = ["multilingual", "science", "cp", "math", "swe"]
# held-out nll of the 98e mlk8-noa at init (train_sft11_mlk8noa.log EVAL init), per-token loss for the share estimate
INIT_NLL = {"agentic": 0.428, "opencode": 0.222, "hermes": 0.243, "kimi31b": 0.377, "ifchat_v1": 0.721,
            "ifchat_v2": 0.657}
BOS, SEQ, PAD = 2, 16384, 0


def segments(f: Path):
    """(name, ids, roles) of every packed example in a render file."""
    for ln in f.open():
        w = json.loads(ln)
        ids, role, names = w["ids"], np.asarray(w["role"], np.int8), w["id"].split("+")
        starts = [i for i, t in enumerate(ids) if t == BOS]
        if len(starts) != len(names) or starts[0] != 0:
            raise SystemExit(f"{f}: window {w['id'][:80]} has {len(starts)} <bos> for {len(names)} names")
        end = len(ids)
        while end > 0 and ids[end - 1] == PAD and role[end - 1] == 0:
            end -= 1
        bounds = [*starts, end]
        for k, name in enumerate(names):
            a, b = bounds[k], bounds[k + 1]
            yield name, ids[a:b], role[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--max-scored", type=int, default=1024)
    ap.add_argument("--ifchat-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opencode-frac", type=float, default=1.0,
                    help="seeded subsample of opencode segments (p50 context 7.6k tokens per ~350 scored: 57%% of the "
                         "windows for 14%% of the loss at 1.0)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists, refusing to overwrite")
    out.mkdir(parents=True)
    sel = {"args": vars(a), "families": {}}
    kept: dict[str, list] = {}
    for fam in KEEP + ANCHOR:
        f = src / f"{fam}.train.roles.jsonl"
        n_seg = n_long = 0
        ends_eog = 0
        keep = []
        for name, ids, role in segments(f):
            n_seg += 1
            scored = int((role == 1).sum())
            if scored > a.max_scored:
                n_long += 1
                continue
            last = int(np.nonzero(role == 1)[0][-1]) if scored else -1
            ends_eog += int(last >= 0 and ids[last] == 106)
            keep.append((name, ids, role, scored))
        if fam == "opencode" and a.opencode_frac < 1.0:
            random.Random(a.seed + 1).shuffle(keep)
            keep = keep[: round(len(keep) * a.opencode_frac)]
        kept[fam] = keep
        sel["families"][fam] = {"segments": n_seg, "dropped_long": n_long, "kept": len(keep),
                                "kept_scored": sum(k[3] for k in keep), "kept_ends_turn": ends_eog,
                                "src_sha256": json.load(open(f.with_suffix(".stats.json")))["sha256"]}
    # ifchat anchor: subsample to --ifchat-share of the estimated loss
    core = sum(INIT_NLL[f] * sel["families"][f]["kept_scored"] for f in KEEP)
    anchor_full = sum(INIT_NLL[f] * sel["families"][f]["kept_scored"] for f in ANCHOR)
    want = a.ifchat_share / (1 - a.ifchat_share) * core
    frac = min(1.0, want / anchor_full) if anchor_full else 0.0
    rng = random.Random(a.seed)
    for fam in ANCHOR:
        k = kept[fam]
        rng.shuffle(k)
        kept[fam] = k[: round(len(k) * frac)]
        sel["families"][fam]["anchor_fraction"] = round(frac, 4)
    total_loss = 0.0
    for fam in KEEP + ANCHOR:
        pk = Packer(SEQ, PAD)
        rows, tbr = [], [0] * N_ROLES

        def emit(done, rows=rows, tbr=tbr):
            for name, x, r in done:
                rows.append({"id": name, "ids": x, "role": r.tolist()})
                for q in range(N_ROLES):
                    tbr[q] += int((r == q).sum())

        for name, ids, role, _ in kept[fam]:
            emit(pk.add(name, ids, role))
        emit(pk.flush())
        p = out / f"{fam}.train.roles.jsonl"
        h = hashlib.sha256()
        with p.open("w") as fh:
            for r in rows:
                s = json.dumps(r) + "\n"
                fh.write(s)
                h.update(s.encode())
        stats = {"rows": len(kept[fam]), "segments": len(kept[fam]), "windows": len(rows), "tokens_by_role": tbr,
                 "sha256": h.hexdigest(), "complete": True,
                 "args": {"selected_from": str(src / f"{fam}.train.roles.jsonl"), "max_scored": a.max_scored,
                          "seed": a.seed, "tool": "sft12_select.py", "seq": SEQ, "pack": True}}
        (out / f"{fam}.train.roles.stats.json").write_text(json.dumps(stats, indent=1))
        sel["families"][fam].update(final_segments=len(kept[fam]), windows=len(rows), scored=tbr[1])
        total_loss += INIT_NLL[fam] * tbr[1]
    for fam in KEEP + ANCHOR:
        sel["families"][fam]["est_loss_share"] = round(INIT_NLL[fam] * sel["families"][fam]["scored"] / total_loss, 4)
    for e in sorted(src.glob("*.eval.roles*")):
        os.link(e, out / e.name)
    sel["dropped_families"] = DROP
    sel["windows"] = sum(v["windows"] for v in sel["families"].values())
    sel["scored"] = sum(v["scored"] for v in sel["families"].values())
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print("%-10s %6s %6s %6s %7s %9s %6s %6s" % ("family", "segs", "long", "kept", "windows", "scored", "share",
                                              "eog%"))
    for fam, v in sel["families"].items():
        print("%-10s %6d %6d %6d %7d %9d %5.1f%% %5.1f%%" % (fam, v["segments"], v["dropped_long"], v["final_segments"],
              v["windows"], v["scored"], 100 * v["est_loss_share"], 100 * v["kept_ends_turn"] / max(v["kept"], 1)))
    print("SFT12_SELECT_OK windows", sel["windows"], "steps(bs2)", sel["windows"] // 2, "scored", sel["scored"])


if __name__ == "__main__":
    main()
