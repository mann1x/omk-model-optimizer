#!/usr/bin/env python3
"""Keep the stage-2 prompt sets (bm-879) before the raw Nemotron download is trimmed.

Whole: RL-Super rlhf.jsonl, RL-Ultra rlhf.jsonl + ifbench.jsonl, both fill_placeholders.py, every README.md.
Sampled: --n seeded random rows (full line-offset scan) of each RLVR file. Out: ROOT/data/stage2_prompts/ plus
manifest.json with upstream repo sha, sha256 of every output and the agent_ref (verifier) census per file.
"""
import argparse
import hashlib
import json
import mmap
import random
import shutil
from collections import Counter
from pathlib import Path

WHOLE = ["RL-Super-Training-Blends/rlhf.jsonl", "RL-Ultra-Training-Blends/rlhf.jsonl",
         "RL-Ultra-Training-Blends/ifbench.jsonl", "RL-Super-Training-Blends/fill_placeholders.py",
         "RL-Ultra-Training-Blends/fill_placeholders.py"]
SAMPLED = ["RL-Super-Training-Blends/rlvr1.jsonl", "RL-Super-Training-Blends/rlvr2.jsonl",
           "RL-Super-Training-Blends/rlvr3.jsonl", "RL-Ultra-Training-Blends/rlvr1.jsonl",
           "RL-Ultra-Training-Blends/rlvr2.jsonl"]


def offsets(p: Path) -> list[int]:
    out, pos = [], 0
    with open(p, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        while pos < len(m):
            out.append(pos)
            j = m.find(b"\n", pos)
            pos = len(m) if j < 0 else j + 1
    return out


def census(p: Path) -> dict:
    c = Counter()
    with open(p) as f:
        for line in f:
            if line.strip() and p.suffix == ".jsonl":
                c[(json.loads(line).get("agent_ref") or {}).get("name", "?")] += 1
    return dict(c.most_common())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/mnt/sdc/omk-model-optimizer"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    nem, out = a.root / "data" / "nemotron", a.root / "data" / "stage2_prompts"
    out.mkdir(parents=True, exist_ok=True)
    up = json.loads((nem / "manifest.json").read_text())
    man = {"seed": a.seed, "n": a.n, "files": {}}
    for f in WHOLE + [f"{d.name}/README.md" for d in sorted(nem.iterdir()) if (d / "README.md").exists()]:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nem / f, dst)
        man["files"][f] = {"mode": "whole"}
    for f in SAMPLED:
        offs = offsets(nem / f)
        pick = sorted(random.Random(f"{a.seed}:{f}").sample(range(len(offs)), min(a.n, len(offs))))
        dst = out / f.replace(".jsonl", f".sample{a.n}.jsonl")
        with open(nem / f, "rb") as src, open(dst, "wb") as o:
            for i in pick:
                src.seek(offs[i])
                o.write(src.readline())
        man["files"][f] = {"mode": "sampled", "rows_total": len(offs), "rows_kept": len(pick), "out": dst.name}
        print(f"{f}: kept {len(pick)}/{len(offs)}", flush=True)
    for f, rec in man["files"].items():
        p = out / (f if rec["mode"] == "whole" else str(Path(f).parent / rec["out"]))
        rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        rec["upstream_sha"] = up.get(f.split("/")[0], {}).get("sha")
        if p.suffix == ".jsonl":
            rec["agent_ref"] = census(p)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("STAGE2_PROMPTS_DONE", flush=True)


if __name__ == "__main__":
    main()
