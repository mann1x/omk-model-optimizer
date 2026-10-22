#!/usr/bin/env python3
"""Stage the omk_qad M6/M7 assets under one bs2 root (bm-875).

Everything lands in ROOT (default /mnt/sdc/omk-model-optimizer):
  models/<repo>                  whole HF model repos, sha256-checked (LFS oid)
  data/fineweb/...               FineWeb-edu shard(s), sha256-checked
  data/nemotron/<set>/<path>     whole Nemotron files, sha256-checked
  data/nemotron/<set>/<path>.windows.jsonl
                                 evenly spaced byte windows of a huge JSONL, trimmed to whole lines
  data/nemotron/manifest.json    what was fetched, with repo sha, sizes, sha256, window offsets

Full files go through aria2c (-c, resumable). Windows go through curl -r and are skipped when their
part file exists. The gated Post-Training-v2 goes through `hf download`, which uses the host's own
login: no token is read or passed here. Stdlib only.
"""
import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
GB = 1024**3
MB = 1024**2

MODELS = ["Qwen/Qwen3.5-4B"]
FINEWEB = ("HuggingFaceFW/fineweb-edu", ["sample/10BT/000_00000.parquet"])
FULL = [
    "RL-Super-Training-Blends", "RL-Ultra-Training-Blends", "SFT-Agentic-v2",
    "SFT-Instruction-Following-Chat-v2", "Instruction-Following-Chat-v1", "SFT-SWE-v2",
    "SFT-SWE-v3.5", "Science-v1", "SFT-OpenCode-v1",
]
# set -> (windows per file, window bytes); small files of a sampled set are taken whole
WINDOWED = {
    "SFT-Math-v3": (16, 512 * MB),
    "Competitive-Programming-v1": (4, 512 * MB),
    "SFT-Competitive-Programming-v2": (4, 512 * MB),
    "SFT-Multilingual-v1": (4, 128 * MB),
}
WHOLE_BELOW = 2 * GB
GATED = ("Post-Training-Dataset-v2",
         ["data/chat-*", "data/code-*", "data/math-*", "data/stem-*", "data/multilingual-*",
          "data/multilingual_de-00000-*", "data/multilingual_es-00000-*", "data/multilingual_fr-00000-*",
          "data/multilingual_it-00000-*", "data/multilingual_ja-00000-*", "README.md"])


def api(kind, rid, rev=None):
    rv = f"/revision/{rev}" if rev else ""
    with urllib.request.urlopen(f"{HF}/api/{kind}/{rid}{rv}?blobs=true", timeout=60) as r:
        return json.load(r)


def url(kind, rid, path, rev):
    pre = "datasets/" if kind == "datasets" else ""
    return f"{HF}/{pre}{rid}/resolve/{rev}/{path}"


def aria(entries, log):
    """entries: (url, dir, out, sha256|None). One aria2c run, resumable, checksums verified."""
    if not entries:
        return
    lst = log.with_suffix(".aria2.txt")
    with open(lst, "w") as f:
        for u, d, out, sha in entries:
            f.write(f"{u}\n  dir={d}\n  out={out}\n")
            if sha:
                f.write(f"  checksum=sha-256={sha}\n")
    cmd = ["aria2c", "-i", str(lst), "-c", "-x8", "-s8", "-j4", "--disable-ipv6=true",
           "--check-integrity=true", "--auto-file-renaming=false", "--allow-overwrite=false",
           "--console-log-level=warn", "--summary-interval=60", f"--log={log}", "--log-level=notice",
           "--max-tries=20", "--retry-wait=10"]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def window(u, size, k, nwin, wbytes, part):
    """Fetch window k of nwin evenly spaced over [0, size); keep whole lines only."""
    if part.exists():
        return json.loads(part.with_suffix(".meta").read_text())
    start = 0 if nwin == 1 else (size - wbytes) * k // (nwin - 1)
    end = min(size, start + wbytes) - 1
    tmp = part.with_suffix(".raw")
    subprocess.run(["curl", "-4", "-sSfL", "--retry", "10", "-r", f"{start}-{end}", "-o", str(tmp), u],
                   check=True)
    raw = tmp.read_bytes()
    if start > 0:  # drop the partial first line
        raw = raw[raw.index(b"\n") + 1:]
    if end + 1 < size:  # drop the partial last line
        raw = raw[: raw.rindex(b"\n") + 1]
    part.write_bytes(raw)
    tmp.unlink()
    meta = {"start": start, "end": end, "kept_bytes": len(raw), "lines": raw.count(b"\n"),
            "sha256": hashlib.sha256(raw).hexdigest()}
    part.with_suffix(".meta").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/sdc/omk-model-optimizer"))
    ap.add_argument("--only", default="models,fineweb,full,windowed,gated")
    ap.add_argument("--sets", default="", help="full/windowed: only these Nemotron sets (comma list; default all)")
    ap.add_argument("--files", default="", help="full: only files matching these globs (comma list; default all)")
    a = ap.parse_args()
    only = set(a.only.split(","))
    sets = set(filter(None, a.sets.split(",")))
    globs = [g for g in a.files.split(",") if g]
    logs = a.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    nem = a.root / "data" / "nemotron"
    man_p = nem / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}

    def info(n):
        """A set already in the manifest is re-fetched at ITS revision: row offsets recorded by build_mix stay valid."""
        rid = f"nvidia/Nemotron-{n}"
        return rid, api("datasets", rid, (man.get(n) or {}).get("sha"))

    def wanted(f):
        return not globs or any(fnmatch.fnmatch(f, g) for g in globs)

    def save():
        nem.mkdir(parents=True, exist_ok=True)
        man_p.write_text(json.dumps(man, indent=1))

    if "models" in only:
        for rid in MODELS:
            d = api("models", rid)
            dst = a.root / "models" / rid.split("/")[1]
            ent = [(url("models", rid, s["rfilename"], d["sha"]), str(dst), s["rfilename"],
                    (s.get("lfs") or {}).get("sha256")) for s in d["siblings"]]
            aria(ent, logs / f"model_{rid.split('/')[1]}.log")
            (dst / "SOURCE.json").write_text(json.dumps({"repo": rid, "sha": d["sha"]}))
    if "fineweb" in only:
        rid, files = FINEWEB
        d = api("datasets", rid)
        sib = {s["rfilename"]: s for s in d["siblings"]} if d.get("siblings") else {}
        ent = []
        for f in files:
            s = sib.get(f) or {}
            ent.append((url("datasets", rid, f, d["sha"]), str(a.root / "data" / "fineweb"),
                        f.replace("/", "_"), (s.get("lfs") or {}).get("sha256")))
        aria(ent, logs / "fineweb.log")
        (a.root / "data" / "fineweb" / "SOURCE.json").write_text(
            json.dumps({"repo": rid, "sha": d["sha"], "files": files}))
    if "full" in only:
        ent = []
        for n in FULL:
            if sets and n not in sets:
                continue
            rid, d = info(n)
            for s in d["siblings"]:
                if wanted(s["rfilename"]):
                    ent.append((url("datasets", rid, s["rfilename"], d["sha"]), str(nem / n), s["rfilename"],
                                (s.get("lfs") or {}).get("sha256")))
            man[n] = {"repo": rid, "sha": d["sha"], "mode": "full",
                      "files": {s["rfilename"]: s.get("size") for s in d["siblings"]}}
        save()
        aria(ent, logs / "nemotron_full.log")
    if "windowed" in only:
        for n, (nwin, wbytes) in WINDOWED.items():
            if sets and n not in sets:
                continue
            rid, d = info(n)
            rec = man.setdefault(n, {"repo": rid, "sha": d["sha"], "mode": "windowed", "files": {}})
            small = []
            for s in d["siblings"]:
                f, size = s["rfilename"], s.get("size") or 0
                u = url("datasets", rid, f, d["sha"])
                if size < WHOLE_BELOW or not f.endswith(".jsonl"):
                    small.append((u, str(nem / n), f, (s.get("lfs") or {}).get("sha256")))
                    rec["files"][f] = {"mode": "full", "size": size}
                    continue
                out = nem / n / (f + ".windows.jsonl")
                out.parent.mkdir(parents=True, exist_ok=True)
                metas = []
                for k in range(nwin):
                    part = out.parent / f"{Path(f).name}.w{k:02d}.part"
                    metas.append(window(u, size, k, nwin, wbytes, part))
                    print(f"   {n}/{f} window {k + 1}/{nwin} lines={metas[-1]['lines']}", flush=True)
                with open(out, "wb") as o:
                    for k in range(nwin):
                        o.write((out.parent / f"{Path(f).name}.w{k:02d}.part").read_bytes())
                rec["files"][f] = {"mode": "windowed", "size": size, "windows": metas}
                save()
            aria(small, logs / f"nemotron_{n}_small.log")
            save()
    if "gated" in only:
        n, pats = GATED
        rid = f"nvidia/Nemotron-{n}"
        d = api("datasets", rid)
        cmd = ["hf", "download", rid, "--repo-type", "dataset", "--revision", d["sha"],
               "--local-dir", str(nem / n)]
        for p in pats:
            cmd += ["--include", p]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        man[n] = {"repo": rid, "sha": d["sha"], "mode": "include", "include": pats}
        save()
    print("STAGE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
