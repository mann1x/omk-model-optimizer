#!/usr/bin/env python3
"""bm-889 sft2: census + fetch of Nemotron PROMPT rows by HTTP range at the pinned revisions (no bulk download).

The responses are NOT trained on (owner 2026-10-29: they will be regenerated); each row is kept whole because some
sets carry their reference answer only inside the response, which the regeneration needs to verify its own answers.

Sampling: a seeded random byte offset in a file, then the first whole row AFTER it. The chance of picking a row is
the length of the row before it, not its own, so the pick is close to row-uniform without a rejection step. Files of
a source get rows in proportion to their estimated row count (size / mean row bytes from a pilot of PILOT rows).
Rows used by sft1 (data/mix_v1, mix_v2 rowmap.jsonl) are excluded by (repo, path, offset) and by sha256.

Out (OUT dir): census.json (per file: size, pilot mean bytes, est rows, top-level keys, roles, tools, answer-like
fields), rows/<source>.jsonl (one line per row: source, repo, rev, path, offset, length, sha256, row), FETCH_OK.
Resumable: offsets are a fixed seeded sequence per file; rows already in rows/<source>.jsonl are skipped.

Usage: sft2_fetch.py OUT [--per-source N] [--pilot K] [--only SRC,SRC] [--census-only]
"""
import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sft_build"))
from build_mix import http_range, line_at  # noqa: E402  (the range reader that fetched mix_v1/v2)

ROOT = Path("/mnt/sdc/omk-model-optimizer")
PINS = ROOT / "runs/m6s2/nemotron_manifest.before_s2.json"
HF = "https://huggingface.co"
# source -> (Nemotron set, file paths); every path is read at the set's pinned sha
SOURCES = {
    "agentic": ("SFT-Agentic-v2", ["data/tool_calling.jsonl", "data/interactive_agent.jsonl", "data/search.jsonl"]),
    "opencode": ("SFT-OpenCode-v1", None),  # None: every data.jsonl of the set
    "swe": ("SFT-SWE-v2", ["data/agentless.jsonl", "data/swe.jsonl"]),
    "ifchat_v2": ("SFT-Instruction-Following-Chat-v2", ["data/reasoning_on.jsonl", "data/reasoning_off.jsonl"]),
    "ifchat_v1": ("Instruction-Following-Chat-v1", ["data/chat_if.jsonl", "data/structured_outputs.jsonl"]),
    "math": ("SFT-Math-v3", ["data/train.jsonl"]),
    "science": ("Science-v1", ["data/MCQ.jsonl", "data/RQA.jsonl"]),
    "cp": ("SFT-Competitive-Programming-v2", None),
    "multilingual": ("SFT-Multilingual-v1", None),
}
ANSWERISH = ("answer", "expected", "solution", "label", "gold", "reference", "verif", "test")


def files_of(pins: dict, set_name: str, paths) -> tuple[str, list[tuple[str, int]]]:
    rec = pins[set_name]
    sizes = {p: (v["size"] if isinstance(v, dict) else v) for p, v in rec["files"].items()}
    if paths is None:
        paths = sorted(p for p in sizes if p.endswith(".jsonl"))
    return rec["sha"], [(p, sizes[p]) for p in paths]


def excluded() -> tuple[set, set]:
    keys, shas = set(), set()
    for m in ("mix_v1", "mix_v2"):
        for line in open(ROOT / "data" / m / "rowmap.jsonl"):
            r = json.loads(line)
            keys.add((r["repo"], r["path"], r["offset"]))
            shas.add(r["sha256"])
    return keys, shas


def row_after(url: str, size: int, off: int) -> tuple[int, bytes] | None:
    """(offset, bytes) of the first whole row starting after byte ``off``; None past the last row."""
    def get(s, e):
        return http_range(url, s, e)
    head = get(off, min(size, off + (64 << 10)) - 1)
    j = head.find(b"\n")
    while j < 0:  # the row under ``off`` is longer than the probe: read on until its end
        more = get(off + len(head), min(size, off + 2 * len(head)) - 1)
        if not more:
            return None
        head += more
        j = head.find(b"\n")
    start = off + j + 1
    if start >= size:
        return None
    return start, line_at(get, start, size, chunk=256 << 10)


def describe(rows: list[dict]) -> dict:
    keys, roles, tools, ans = {}, {}, 0, {}
    for r in rows:
        for k in r:
            keys[k] = keys.get(k, 0) + 1
            if any(a in k.lower() for a in ANSWERISH):
                ans[k] = ans.get(k, 0) + 1
        msgs = r.get("messages") or r.get("conversations") or []
        pat = ",".join((m.get("role") or m.get("from") or "?") for m in msgs[:8]) + (",…" if len(msgs) > 8 else "")
        roles[pat] = roles.get(pat, 0) + 1
        tools += bool(r.get("tools"))
        md = r.get("metadata")
        if isinstance(md, dict):
            for k in md:
                if any(a in k.lower() for a in ANSWERISH):
                    ans["metadata." + k] = ans.get("metadata." + k, 0) + 1
    top = sorted(roles.items(), key=lambda kv: -kv[1])[:5]
    return {"keys": keys, "role_patterns": top, "with_tools": tools, "answer_fields": ans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-source", type=int, default=3000, help="candidate prompt rows per source")
    ap.add_argument("--pilot", type=int, default=8, help="rows per file for the census (mean bytes, schema)")
    ap.add_argument("--only", default="")
    ap.add_argument("--census-only", action="store_true")
    ap.add_argument("--seed", type=int, default=889)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    pins = json.loads(PINS.read_text())
    ex_keys, ex_shas = excluded()
    print(f"EXCLUDE {len(ex_keys)} sft1 rows (mix_v1 + mix_v2 rowmaps)", flush=True)
    only = set(filter(None, a.only.split(",")))
    census_p = out / "census.json"
    census = json.loads(census_p.read_text()) if census_p.exists() else {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(4)  # HF rate-limits range reads (429 at 16 in flight)

    for src, (set_name, paths) in SOURCES.items():
        if only and src not in only:
            continue
        rev, files = files_of(pins, set_name, paths)
        repo = f"nvidia/Nemotron-{set_name}"
        if src not in census:
            per = {}
            for p, size in files:
                url = f"{HF}/datasets/{repo}/resolve/{rev}/{p}"
                rng = random.Random(f"{a.seed}/pilot/{repo}/{p}")
                offs = [rng.randrange(0, size) for _ in range(a.pilot)]
                got = [g for g in pool.map(lambda o, u=url, s=size: row_after(u, s, o), offs) if g]
                rows = [json.loads(b) for _, b in got]
                mean = sum(len(b) for _, b in got) / max(1, len(got))
                per[p] = {"size": size, "pilot_rows": len(got), "mean_bytes": round(mean),
                          "est_rows": round(size / mean) if mean else 0, **describe(rows)}
                print(f"CENSUS {src} {p}: {size / 1e9:.2f} GB, mean row {mean / 1e3:.1f} kB, "
                      f"~{per[p]['est_rows']:,} rows, tools {per[p]['with_tools']}/{len(got)}, "
                      f"answer fields {per[p]['answer_fields']}, roles {per[p]['role_patterns'][:2]}", flush=True)
            census[src] = {"set": set_name, "repo": repo, "rev": rev, "files": per}
            census_p.write_text(json.dumps(census, indent=1))
        if a.census_only:
            continue

        per = census[src]["files"]
        tot = sum(f["est_rows"] for f in per.values()) or 1
        dst = out / "rows" / f"{src}.jsonl"
        have = {}
        seen = set()
        if dst.exists():
            for line in open(dst):
                r = json.loads(line)
                have[r["path"]] = have.get(r["path"], 0) + 1
                seen.add((r["path"], r["offset"]))
        with open(dst, "a") as fh:
            for p, f in per.items():
                want = max(1, round(a.per_source * f["est_rows"] / tot))
                url = f"{HF}/datasets/{repo}/resolve/{rev}/{p}"
                rng = random.Random(f"{a.seed}/fetch/{repo}/{p}")
                n = have.get(p, 0)
                skipped = 0
                while n < want:
                    offs = [rng.randrange(0, f["size"]) for _ in range(min(64, 2 * (want - n)))]
                    for g in pool.map(lambda o, u=url, s=f["size"]: row_after(u, s, o), offs):
                        if not g or n >= want:
                            continue
                        start, b = g
                        sha = hashlib.sha256(b).hexdigest()
                        if (p, start) in seen or (repo, p, start) in ex_keys or sha in ex_shas:
                            skipped += 1
                            continue
                        seen.add((p, start))
                        rec = {"source": src, "repo": repo, "rev": rev, "path": p, "offset": start,
                               "length": len(b), "sha256": sha, "row": json.loads(b)}
                        with lock:
                            fh.write(json.dumps(rec) + "\n")
                        n += 1
                    fh.flush()
                print(f"FETCH {src} {p}: {n}/{want} rows (skipped {skipped}: duplicate or sft1)", flush=True)
    print("FETCH_OK" if not a.census_only else "CENSUS_OK", flush=True)


if __name__ == "__main__":
    main()
