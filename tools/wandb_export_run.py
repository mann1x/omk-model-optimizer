#!/usr/bin/env python3
"""Export a W&B run completely to a local directory, so a training record never depends on W&B alone.

Writes, per run:   <out>/<project>/<run_name>__<run_id>/
    run.json            id, name, project, entity, state, created_at, url, tags, notes, host, git commit, command line
    config.json         the full run config (every trainer argument)
    summary.json        final summary values
    history.jsonl       EVERY logged row with EVERY key (scan_history, no sampling) — one JSON object per row
    system.jsonl        system metrics stream (GPU util / memory / power / network), when available
    files/              every file the run uploaded (output.log = trainer console, config.yaml, requirements.txt,
                        wandb-metadata.json, ...)
    MANIFEST.txt        sizes + sha256 of everything above

Usage:  wandb_export_run.py --out DIR  entity/project/run_id [entity/project/run_id ...]
        wandb_export_run.py --out DIR  --project entity/project          (every run of the project)
Needs WANDB_API_KEY in the environment or ~/.netrc (never on the command line). Read-only against W&B.
"""
import argparse
import hashlib
import json
import os
import sys

import wandb


def sha256(path):
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_run(api, path, out_root):
    """Dump one run (metadata, config, summary, full history, system stream, files, manifest) under out_root."""
    run = api.run(path)
    out = os.path.join(out_root, run.project, f"{run.name}__{run.id}")
    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    meta = {
        "id": run.id, "name": run.name, "project": run.project, "entity": run.entity, "state": run.state,
        "created_at": str(run.created_at), "url": run.url, "tags": list(run.tags), "notes": run.notes,
        "path": "/".join(run.path),
    }
    md = getattr(run, "metadata", None) or {}
    for k in ("host", "git", "program", "args", "codePath", "python", "cuda", "gpu", "gpu_count", "startedAt"):
        if k in md:
            meta[k] = md[k]
    json.dump(meta, open(os.path.join(out, "run.json"), "w"), indent=1, default=str)
    json.dump(dict(run.config), open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
    json.dump(dict(run.summary.items()), open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)

    n = 0
    with open(os.path.join(out, "history.jsonl"), "w") as f:
        for row in run.scan_history(page_size=1000):
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
    ns = 0
    try:
        ev = run.history(stream="events", pandas=False, samples=1_000_000)
        with open(os.path.join(out, "system.jsonl"), "w") as f:
            for row in ev:
                f.write(json.dumps(row, default=str) + "\n")
                ns += 1
    except Exception as e:  # best effort: the events stream is not always served
        open(os.path.join(out, "system.jsonl.UNAVAILABLE"), "w").write(f"{type(e).__name__}: {e}\n")
    nf = 0
    for fobj in run.files():
        if fobj.size == 0 and fobj.name.startswith("artifact/"):
            continue
        try:
            fobj.download(root=os.path.join(out, "files"), replace=True)
            nf += 1
        except Exception as e:
            marker = os.path.join(out, "files", fobj.name.replace("/", "_") + ".FAILED")
            open(marker, "w").write(f"{type(e).__name__}: {e}\n")

    lines = []
    for dp, _, fns in os.walk(out):
        for fn in sorted(fns):
            if fn == "MANIFEST.txt":
                continue
            p = os.path.join(dp, fn)
            lines.append(f"{os.path.getsize(p):>12} {sha256(p)} {os.path.relpath(p, out)}")
    open(os.path.join(out, "MANIFEST.txt"), "w").write("\n".join(lines) + "\n")
    print(f"EXPORTED {path} -> {out}: history rows {n}, system rows {ns}, files {nf}, state {run.state}")
    return out


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", help="entity/project: export every run of it")
    ap.add_argument("runs", nargs="*", help="entity/project/run_id")
    a = ap.parse_args()
    api = wandb.Api(timeout=180)
    paths = list(a.runs)
    if a.project:
        paths += [f"{a.project}/{r.id}" for r in api.runs(a.project)]
    if not paths:
        ap.error("give run paths or --project")
    for p in paths:
        export_run(api, p, a.out)


if __name__ == "__main__":
    sys.exit(main())
