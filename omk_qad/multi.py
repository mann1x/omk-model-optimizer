# SPDX-License-Identifier: Apache-2.0
"""Separate students over one shared teacher (plan M5), e.g. an IQ2_XXS and a ternary student side by side.

Shared, computed once, because it cannot change a student's quality:
  - the layout probe;
  - the teacher top-K cache over the training stream, and the eval rows;
  - the imatrix pass (model-basis and folded statistics).
Separate, per student: layout, latents, optimizer, LR / schedule / loss / ternarizer, checkpoints and export.
Each student trains in its own ``omk_qad.train`` process, so it pauses, resumes, fails and is relaunched on its own.
The spec places each student on a GPU; several may share one once ``preflight`` has measured that they fit.

    python -m omk_qad.multi --spec run.yaml plan       # probe once; one layout per student; block check
    python -m omk_qad.multi --spec run.yaml cache      # teacher cache + imatrix, once
    python -m omk_qad.multi --spec run.yaml preflight  # each student alone for a few steps: peak memory, fit check
    python -m omk_qad.multi --spec run.yaml train      # one process per student, concurrently (resumable)
    python -m omk_qad.multi --spec run.yaml export     # one GGUF per finished student
    python -m omk_qad.multi --spec run.yaml report     # one table over the same eval rows
    python -m omk_qad.multi --spec run.yaml all        # plan, cache, train, export, report

Spec (YAML or JSON):

    model: /models/Qwen3.5-4B
    out: /runs/qwen35_4b_smoke            # <out>/cache, <out>/<student>/..., <out>/logs
    llama_cpp: /opt/llama.cpp
    layout: {rest: IQ2_M}                 # defaults of every student's plan: rest, mtp, mtp_type
    cache: {gpu: 1, data: corpus.jsonl, eval_data: eval.jsonl, n_seq: 16000, n_eval: 32, seq: 4096}
    preflight_steps: 3
    students:
      - name: iq2
        gpu: 1
        layout: {fixed: {dense_mlp: iq2_xxs}}
        train: {lr: 1.0e-5, loss: jaccard, steps: 2000, bs: 8}
      - name: ternary
        gpu: 1
        layout: {fixed: {dense_mlp: pq2_0}, override: {attention: pq2_0, linear_attn: pq2_0, lm_head: pq2_0}}
        train: {lr: 6.0e-5, loss: kl, steps: 2000, bs: 8, ternary_method: "top:86"}
    references:                           # optional notes under the table (other runs: not table rows)
      bm860_r1: {note: "full-vocab KL code 0.732 / ..."}

``gpu`` is a physical GPU index (the process sees only that GPU) or ``null`` for CPU. ``cache`` / ``train`` keys
are ``omk_qad.train`` flags without the dashes (``n_seq`` = ``--n-seq``); ``true`` passes a bare flag. A student's
``layout`` may also be the path of a saved layout (.pt).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

OWNED = {"phase", "model", "layout", "cache", "out", "device", "llama_cpp", "gguf", "load"}  # set by this driver
PHASES = ("plan", "cache", "preflight", "train", "export", "report")
CTX_GIB = 1.0  # per-process CUDA context + allocator slack on top of the reserved peak


def load_spec(path: Path) -> dict:
    """Read and validate a spec."""
    text = Path(path).read_text()
    if path.suffix == ".json":
        spec = json.loads(text)
    else:
        import yaml

        spec = yaml.safe_load(text)
    for k in ("model", "out", "students"):
        if k not in spec:
            raise SystemExit(f"spec lacks {k!r}")
    names = [s["name"] for s in spec["students"]]
    if len(set(names)) != len(names) or not names:
        raise SystemExit(f"student names must be unique and non-empty: {names}")
    for s in spec["students"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", s["name"]) or s["name"] in ("cache", "logs", "probe", "_preflight"):
            raise SystemExit(f"invalid student name {s['name']!r}")
        clash = OWNED & set(s.get("train", {}))
        if clash:
            raise SystemExit(f"student {s['name']}: {sorted(clash)} are set by omk_qad.multi, not in 'train'")
    if OWNED & set(spec.get("cache", {})) - {"layout"}:
        raise SystemExit(f"cache: {sorted(OWNED & set(spec['cache']))} are set by omk_qad.multi")
    spec.setdefault("llama_cpp", "/opt/llama.cpp")
    spec["out"] = str(Path(spec["out"]))
    return spec


def _flags(d: dict) -> list[str]:
    out = []
    for k, v in d.items():
        if k == "gpu" or v is None or v is False:
            continue
        out.append("--" + k.replace("_", "-"))
        if v is not True:
            out.append(str(v))
    return out


def _env(gpu) -> tuple[dict, str]:
    """Environment + --device for a physical GPU index (or None = CPU)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
    return env, "cpu" if gpu is None else "cuda:0"


def _paths(spec: dict, s: dict | None = None) -> dict:
    out = Path(spec["out"])
    p = {"out": out, "cache": out / "cache", "logs": out / "logs", "probe": out / "probe", "state": out / "multi.json"}
    if s is not None:
        d = out / s["name"]
        p.update(dir=d, layout=d / "layout.pt", final=d / "stage1" / "final.pt", summary=d / "stage1" / "summary.json",
                 gguf=d / f"{s['name']}.gguf", preflight=out / "_preflight" / s["name"])
    return p


def _students(spec: dict, only: str) -> list[dict]:
    if not only:
        return spec["students"]
    want = only.split(",")
    unknown = set(want) - {s["name"] for s in spec["students"]}
    if unknown:
        raise SystemExit(f"unknown students {sorted(unknown)}")
    return [s for s in spec["students"] if s["name"] in want]


def _state(spec: dict) -> dict:
    p = _paths(spec)["state"]
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(spec: dict, st: dict) -> None:
    p = _paths(spec)["state"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".tmp").write_text(json.dumps(st, indent=1))
    p.with_suffix(".tmp").replace(p)


def _same_plan(a: dict, b: dict) -> bool:
    return {g: t["type"] for g, t in a["targets"].items()} == {g: t["type"] for g, t in b["targets"].items()} \
        and a.get("hadamard_block") == b.get("hadamard_block")


# ---------------------------------------------------------------------------------------------------------- plan
def phase_plan(spec: dict, students: list[dict]) -> None:
    """One probe; one layout per student (an existing layout.pt is never replaced); the shared-cache block check."""
    from .layout import plan_layout, probe_model, summary

    base = spec.get("layout", {})
    probe = None
    try:
        for s in students:
            p = _paths(spec, s)
            p["dir"].mkdir(parents=True, exist_ok=True)
            src = s.get("layout", {})
            if isinstance(src, str):
                lay = torch.load(src, weights_only=False)
            else:
                cfg = {**base, **src}
                if probe is None:
                    probe = probe_model(Path(spec["model"]), Path(spec["llama_cpp"]), _paths(spec)["probe"],
                                        cfg.get("mtp", True))
                kw = {"fixed": cfg.get("fixed"), "overrides": cfg.get("override"), "regex": cfg.get("regex")}
                if "mtp_type" in cfg:
                    kw["mtp_type"] = cfg["mtp_type"]
                lay = plan_layout(probe, cfg.get("rest", "IQ2_M"), **kw)
            if p["layout"].exists():
                old = torch.load(p["layout"], weights_only=False)
                if not _same_plan(old, lay):
                    raise SystemExit(f"student {s['name']}: the spec's layout differs from its existing "
                                     f"{p['layout']} (a run's layout is fixed; use a new student name)")
                print(f"plan {s['name']}: {p['layout']} kept (same plan)")
                continue
            torch.save(lay, p["layout"])
            sm = summary(lay)
            p["layout"].with_suffix(".json").write_text(json.dumps(sm, indent=1))
            print(f"plan {s['name']}: {sm['type_counts']} block {sm['hadamard_block']} "
                  f"{sm['sizes_mib'].get('quant')} MiB -> {p['layout']}")
    finally:
        if probe is not None:
            probe["sparse"].unlink(missing_ok=True)
    check_blocks(spec)


def cache_layout(spec: dict) -> Path:
    """The layout the shared cache phase is given: the first student's (any plan of the model maps the same names)."""
    return _paths(spec, spec["students"][0])["layout"]


def check_blocks(spec: dict) -> int | None:
    """Every ternary student must fold with the block the shared folded imatrix is computed in."""
    from .imatrix import folded_candidates

    lay0 = cache_layout(spec)
    if not lay0.exists():
        raise SystemExit(f"{lay0} missing: run 'plan' first")
    _, block = folded_candidates(torch.load(lay0, weights_only=False))
    for s in spec["students"]:
        p = _paths(spec, s)["layout"]
        if not p.exists():
            raise SystemExit(f"{p} missing: run 'plan' first")
        b = torch.load(p, weights_only=False).get("hadamard_block")
        if b is not None and b != block:
            raise SystemExit(f"student {s['name']} folds with block {b}, the shared folded imatrix with {block} "
                             "(the largest block dividing every foldable width): give it the same ternary classes' "
                             "widths, or train it from its own cache")
    return block


# --------------------------------------------------------------------------------------------------------- cache
def phase_cache(spec: dict) -> None:
    """The shared teacher cache + imatrix (resumable: omk_qad.train's cache phase continues where it stopped)."""
    check_blocks(spec)
    p = _paths(spec)
    p["logs"].mkdir(parents=True, exist_ok=True)
    c = dict(spec.get("cache", {}))
    env, dev = _env(c.get("gpu"))
    cmd = [sys.executable, "-m", "omk_qad.train", "--phase", "cache", "--model", spec["model"], "--cache",
           str(p["cache"]), "--layout", str(cache_layout(spec)), "--device", dev, "--llama-cpp", spec["llama_cpp"],
           *_flags(c)]
    with open(p["logs"] / "cache.out", "a") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
    if rc or not (p["cache"] / "cache.json").exists():
        raise SystemExit(f"cache phase failed (rc {rc}): see {p['logs'] / 'cache.out'}")
    print(f"cache: {p['cache']}")


# ------------------------------------------------------------------------------------------------ train processes
def _train_cmd(spec: dict, s: dict, out: Path, overrides: dict | None = None) -> tuple[list[str], dict]:
    env, dev = _env(s.get("gpu"))
    p = _paths(spec, s)
    args = {**s.get("train", {}), **(overrides or {})}
    cmd = [sys.executable, "-m", "omk_qad.train", "--phase", "train", "--model", spec["model"], "--layout",
           str(p["layout"]), "--cache", str(_paths(spec)["cache"]), "--out", str(out), "--device", dev,
           "--llama-cpp", spec["llama_cpp"], *_flags(args)]
    return cmd, env


def _launch(jobs: list[tuple[str, list[str], dict, Path]]) -> dict:
    """Start every job at once, wait for all; {name: {pid, rc, started, ended, log}}."""
    procs, out = {}, {}
    for name, cmd, env, log in jobs:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a")
        procs[name] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
        out[name] = {"pid": procs[name][0].pid, "started": time.time(), "cmd": cmd, "log": str(log)}
        print(f"started {name}: pid {out[name]['pid']} (log {log})")
    running = dict(procs)
    while running:  # poll, so each student's end time is its own
        for name, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                out[name]["rc"], out[name]["ended"] = proc.returncode, time.time()
                fh.close()
                del running[name]
        if running:
            time.sleep(1.0)
    return out


def _peak(log: Path) -> dict:
    m = re.findall(r"PEAK_MEM allocated=([\d.]+)GiB reserved=([\d.]+)GiB", log.read_text() if log.exists() else "")
    if not m:
        return {}
    return {"allocated_gib": float(m[-1][0]), "reserved_gib": float(m[-1][1])}


def _free_gib() -> dict[int, float]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    return {int(i): float(f) / 1024 for i, f in (line.split(",") for line in q.strip().splitlines())}


def phase_preflight(spec: dict, students: list[dict]) -> None:
    """Each student ALONE for a few steps (its own scratch run): peak memory, then the co-scheduling fit check."""
    check_blocks(spec)
    n = int(spec.get("preflight_steps", 3))
    st = _state(spec)
    pf = st.setdefault("preflight", {})
    for s in students:
        p = _paths(spec, s)
        if p["preflight"].exists():
            shutil.rmtree(p["preflight"])  # the preflight's own scratch run, never a student run
        cmd, env = _train_cmd(spec, s, p["preflight"], {"steps": n, "eval_every": n, "save_every": n,
                                                        "snapshot_every": 0, "no_wandb": True})
        res = _launch([(s["name"], cmd, env, p["preflight"] / "train.out")])[s["name"]]
        if res["rc"]:
            raise SystemExit(f"preflight of {s['name']} failed (rc {res['rc']}): see {res['log']}")
        pf[s["name"]] = {"gpu": s.get("gpu"), "steps": n, **_peak(p["preflight"] / "stage1" / "log.txt")}
        for ck in (p["preflight"] / "stage1").glob("*.pt"):  # a full student checkpoint per preflight: never reused
            ck.unlink()
        print(f"preflight {s['name']}: {pf[s['name']]}")
    _save_state(spec, st)
    fit(spec, spec["students"], st)


def fit(spec: dict, students: list[dict], st: dict | None = None) -> None:
    """Refuse to co-schedule students on a GPU unless each one's measured peak sums within its free memory."""
    st = _state(spec) if st is None else st
    by_gpu: dict[int, list[str]] = {}
    for s in students:
        if s.get("gpu") is not None:
            by_gpu.setdefault(int(s["gpu"]), []).append(s["name"])
    shared = {g: names for g, names in by_gpu.items() if len(names) > 1}
    if not shared:
        print("fit: no GPU is shared" + (" (CPU students: no memory check)" if len(by_gpu) < len(students) else ""))
        return
    pf = st.get("preflight", {})
    missing = [n for names in shared.values() for n in names if "reserved_gib" not in pf.get(n, {})]
    if missing:
        raise SystemExit(f"students {missing} share a GPU without a measured peak: run 'preflight' first "
                         "(co-scheduled students have OOM'd before, bug-759)")
    free = _free_gib()
    for g, names in shared.items():
        need = sum(pf[n]["reserved_gib"] + CTX_GIB for n in names)
        print(f"fit GPU {g}: {names} need {need:.1f} GiB, free {free[g]:.1f} GiB")
        if need > free[g]:
            raise SystemExit(f"GPU {g}: {names} need {need:.1f} GiB > {free[g]:.1f} GiB free; move a student")


def phase_train(spec: dict, students: list[dict]) -> int:
    """One omk_qad.train process per unfinished student, all at once; each resumes from its own last.pt."""
    check_blocks(spec)
    st = _state(spec)
    todo = []
    for s in students:
        p = _paths(spec, s)
        if p["final"].exists():
            print(f"train {s['name']}: stage 1 finished ({p['final']}), not relaunched")
            continue
        todo.append(s)
    fit(spec, todo, st)
    jobs = []
    for s in todo:
        p = _paths(spec, s)
        cmd, env = _train_cmd(spec, s, p["dir"])
        jobs.append((s["name"], cmd, env, _paths(spec)["logs"] / f"{s['name']}.train.out"))
    res = _launch(jobs)
    st.setdefault("train", {}).update(res)
    _save_state(spec, st)
    bad = {n: r["rc"] for n, r in res.items() if r["rc"]}
    for n, r in res.items():
        print(f"train {n}: rc {r['rc']} in {r['ended'] - r['started']:.0f}s")
    if bad:
        print(f"FAILED students {bad}: see their logs; relaunching resumes each from its last.pt")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------------------- export, report
def phase_export(spec: dict, students: list[dict]) -> int:
    """One GGUF per finished student, at its own layout, with its own imatrix (as it trained)."""
    rc = 0
    for s in students:
        p = _paths(spec, s)
        if not p["final"].exists():
            print(f"export {s['name']}: NOT exported, stage 1 not finished ({p['final']} missing)")
            rc = 1
            continue
        env, dev = _env(s.get("gpu"))
        cmd = [sys.executable, "-m", "omk_qad.train", "--phase", "export", "--model", spec["model"], "--layout",
               str(p["layout"]), "--device", dev, "--load", str(p["final"]), "--gguf", str(p["gguf"]),
               "--llama-cpp", spec["llama_cpp"]]
        for k in ("latent_dtype", "latent_min", "export_only", "ternary_method"):
            if k in s.get("train", {}):
                cmd += _flags({k: s["train"][k]})
        with open(_paths(spec)["logs"] / f"{s['name']}.export.out", "a") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
        print(f"export {s['name']}: rc {r} -> {p['gguf']}")
        rc |= int(r != 0)
    return rc


COLS = ("klk", "jaccard", "top1", "ppl")


def phase_report(spec: dict, students: list[dict]) -> dict:
    """The students' evals on the shared eval rows (plus spec references) as one table: report.md / report.json."""
    rows, caches = [], set()
    for s in students:
        p = _paths(spec, s)
        if not p["summary"].exists():
            rows.append({"student": s["name"], "status": "no summary.json"})
            continue
        sm = json.loads(p["summary"].read_text())
        caches.add(sm["origin"]["cache"])
        res = sm["results"]
        init = next((r for r in res if r["tag"] == "init"), {})
        last = res[-1]
        lay = json.loads(p["layout"].with_suffix(".json").read_text()) if p["layout"].with_suffix(".json").exists() \
            else {}
        exp = p["gguf"].with_suffix(".export.json")
        row = {"student": s["name"], "status": "final" if p["final"].exists() else "running/paused",
               "at": last["tag"], "lr": sm["args"]["lr"], "loss": sm["args"]["loss"],
               "fixed": lay.get("fixed"), "teacher_ppl": last.get("teacher_ppl"),
               **{c: last.get(c) for c in COLS}, **{f"init_{c}": init.get(c) for c in COLS}}
        for fam, m in last.get("families", {}).items():
            row[f"klk:{fam}"] = m["klk"]
        if exp.exists():
            row["gguf_mib"] = round(json.loads(exp.read_text())["size_bytes"] / 2**20, 1)
        rows.append(row)
    refs = spec.get("references", {})  # other runs: other metric / rows unless proven same; never a table row
    same = len(caches) <= 1
    out = {"rows": rows, "same_eval_rows": same, "caches": sorted(caches), "references": refs}
    fams = sorted({k for r in rows for k in r if k.startswith("klk:")})
    hdr = ["student", "status", "at", "lr", "loss", *COLS, *fams, "init_klk", "gguf_mib"]

    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))

    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(fmt(r.get(h)) for h in hdr) + " |" for r in rows]
    if not same:
        md.append(f"\n**WARNING: students trained on different caches {sorted(caches)}: rows are not comparable.**")
    if refs:
        md.append("\nReferences (not the table's metric or eval rows unless stated):")
        md += [f"- {name}: {json.dumps(ref)}" for name, ref in refs.items()]
    p = _paths(spec)
    (p["out"] / "report.md").write_text("\n".join(md) + "\n")
    (p["out"] / "report.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))
    return out


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=(*PHASES, "all"))
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--students", default="", help="comma list: act on these students only")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    students = _students(spec, a.students)
    Path(spec["out"]).mkdir(parents=True, exist_ok=True)
    rc = 0
    for ph in (("plan", "cache", "train", "export", "report") if a.phase == "all" else (a.phase,)):
        if ph == "plan":
            phase_plan(spec, students)
        elif ph == "cache":
            phase_cache(spec)
        elif ph == "preflight":
            phase_preflight(spec, students)
        elif ph == "train":
            rc = phase_train(spec, students)
            if rc and a.phase == "all":
                break
        elif ph == "export":
            rc |= phase_export(spec, students)
        else:
            phase_report(spec, students)
    sys.exit(rc)


if __name__ == "__main__":
    main()
