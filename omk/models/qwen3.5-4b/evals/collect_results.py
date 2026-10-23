#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect the Qwen3.5-4B M6 eval results from a run dir into results.json + results.md (stdlib only).

    python collect_results.py RUN_DIR [--out DIR]     # RUN_DIR = .../runs/m6_qwen35_4b

Reads RUN_DIR/compare/kld/kld_results.jsonl (gguf_kld rows; opencoti-tool rows are RUN_DIR/he/ockld.<tag>.log)
and every lm-eval RUN_DIR/he/out_<name>[_hep]/<name>/results_*.json (pass@1,create_test). Rows without a
measurement are kept with null, so a missing cell is visible rather than dropped.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

# label, kind, GGUF (relative to RUN_DIR), KLD tag in kld_results.jsonl, served name in he/, engine
ROWS = [
    ("Qwen3.5-4B F16", "stock", "compare/Qwen3.5-4B-F16.gguf", None, "qwen3.5-4b-f16", "llama.cpp"),
    ("Q4_K_M", "stock", "compare/Qwen3.5-4B-Q4_K_M.gguf", "Q4_K_M", "qwen3.5-4b-q4km", "llama.cpp"),
    ("Q4_K_M (opencoti engine control)", "control", "compare/Qwen3.5-4B-Q4_K_M.gguf", "opencoti:q4km",
     "qwen3.5-4b-q4km-oc",
     "opencoti"),
    ("Q4_K_S", "stock", "compare/Qwen3.5-4B-Q4_K_S.gguf", "Q4_K_S", "qwen3.5-4b-q4ks", "llama.cpp"),
    ("Q3_K_L", "stock", "compare/Qwen3.5-4B-Q3_K_L.gguf", "Q3_K_L", "qwen3.5-4b-q3kl", "llama.cpp"),
    ("Q3_K_M", "stock", "compare/Qwen3.5-4B-Q3_K_M.gguf", "Q3_K_M", "qwen3.5-4b-q3km", "llama.cpp"),
    ("Q3_K_S", "stock", "compare/Qwen3.5-4B-Q3_K_S.gguf", "Q3_K_S", "qwen3.5-4b-q3ks", "llama.cpp"),
    ("IQ3_M", "stock", "compare/Qwen3.5-4B-IQ3_M.gguf", "IQ3_M", "qwen3.5-4b-iq3m", "llama.cpp"),
    ("IQ3_XS", "stock", "compare/Qwen3.5-4B-IQ3_XS.gguf", "IQ3_XS", "qwen3.5-4b-iq3xs", "llama.cpp"),
    ("IQ3_XXS", "stock", "compare/Qwen3.5-4B-IQ3_XXS.gguf", "IQ3_XXS", "qwen3.5-4b-iq3xxs", "llama.cpp"),
    ("student IQ2_XXS core, IQ3_M packaging + Q8_0 head", "student", "pack/iq3m_l/iq2-pack-iq3m_l.gguf",
     "M6-iq2-pack-iq3m_l", "m6-iq2-pack-iq3m_l", "llama.cpp"),
    ("student IQ2_XXS core, Q3_K_M packaging", "student", "pack/q3km/iq2-pack-q3km.gguf", "M6-iq2-pack-q3km",
     "m6-iq2-pack-q3km", "llama.cpp"),
    ("student IQ2_XXS core, IQ3_M packaging", "student", "pack/iq3m/iq2-pack-iq3m.gguf", "M6-iq2-pack-iq3m",
     "m6-iq2-pack-iq3m", "llama.cpp"),
    ("IQ2_M", "stock", "compare/Qwen3.5-4B-IQ2_M.gguf", "IQ2_M", "qwen3.5-4b-iq2m", "llama.cpp"),
    ("student MLP re-encoded at IQ2_XS (off its trained grid)", "student-offgrid", "xq/iq2xs/iq2-iq2xs.gguf",
     "M6-iq2@iq2xs", "m6-iq2-student-iq2xs", "llama.cpp"),
    ("IQ2_S", "stock", "compare/Qwen3.5-4B-IQ2_S.gguf", "IQ2_S", "qwen3.5-4b-iq2s", "llama.cpp"),
    ("IQ2_XS", "stock", "compare/Qwen3.5-4B-IQ2_XS.gguf", "IQ2_XS", "qwen3.5-4b-iq2xs", "llama.cpp"),
    ("student IQ2_XXS core, IQ3_XXS packaging", "student", "pack/iq3xxs/iq2-pack-iq3xxs.gguf", "M6-iq2-pack-iq3xxs",
     "m6-iq2-pack-iq3xxs", "llama.cpp"),
    ("**student IQ2_XXS stage 2 (mix_v2, lr 3e-6)**", "student",
     "iq2/stage2/mix_v2_lr3e6/iq2_s2_mix_v2_lr3e6.gguf", "M6-iq2-s2-mix_v2_lr3e6", "m6-iq2-s2-mix_v2_lr3e6",
     "llama.cpp"),
    ("**student IQ2_XXS (as trained, IQ2_M packaging)**", "student", "iq2/iq2.gguf", "M6-iq2", "m6-iq2-student",
     "llama.cpp"),
    ("IQ2_XXS", "stock", "compare/Qwen3.5-4B-IQ2_XXS.gguf", "IQ2_XXS", "qwen3.5-4b-iq2xxs", "llama.cpp"),
    ("student MLP re-encoded at IQ1_M (off its trained grid)", "student-offgrid", "xq/iq1m/iq2-iq1m.gguf",
     "M6-iq2@iq1m", "m6-iq2-student-iq1m", "llama.cpp"),
    ("IQ1_M", "stock", "compare/Qwen3.5-4B-IQ1_M.gguf", "IQ1_M", "qwen3.5-4b-iq1m", "llama.cpp"),
    ("**student ternary (PQ2_0)**", "student", "tern/tern.gguf", "opencoti:tern", "m6-tern-student", "opencoti"),
]


def he_score(run: Path, name: str, plus: bool):
    """pass@1 of one lm-eval humaneval / humaneval_plus run, or None if it was not run."""
    d = run / "he" / (f"out_{name}_hep" if plus else f"out_{name}")
    fs = sorted(glob.glob(str(d / "*" / "results_*.json")))
    if not fs:
        return None
    r = json.load(open(fs[-1]))["results"]
    return r[next(iter(r))]["pass@1,create_test"]


def opencoti_kld(log: Path):
    """KLD row of the PQ2_0 file, parsed from the opencoti perplexity tool's log (stock llama.cpp cannot load it)."""
    if not log.exists():
        return None
    t = log.read_text(errors="replace")
    k = re.search(r"Mean\s+KLD:\s+([\d.]+)", t)
    s = re.search(r"Same top p:\s+([\d.]+)", t)
    p = re.search(r"Mean PPL\(Q\)\s+:\s+([\d.]+)", t)
    if k is None or s is None or p is None:
        return None
    return {"kld_mean": float(k.group(1)), "same_top_p_pct": float(s.group(1)), "ppl_q": float(p.group(1)),
            "tool": "opencoti perplexity kld-2610220831001"}


def main():
    """Write results.json and results.md next to this script (or --out)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    kld = {}
    for line in open(a.run / "compare/kld/kld_results.jsonl"):
        r = json.loads(line)
        kld[r["quant"]] = {"kld_mean": r["kld_mean"][0], "same_top_p_pct": r["same_top_p_pct"][0],
                           "ppl_q": r["ppl_q"][0], "ppl_base": r["ppl_base"][0], "tool": "llama.cpp 311d4211bf"}
    for log in (a.run / "he").glob("ockld.*.log"):  # opencoti-tool rows: ockld.<tag>.log -> "opencoti:<tag>"
        kld["opencoti:" + log.name.split(".")[1]] = opencoti_kld(log)
    ppl_f16 = next((v["ppl_base"] for v in kld.values() if v and "ppl_base" in v), None)
    rows = []
    for label, kind, rel, ktag, he, engine in ROWS:
        g = a.run / rel
        rows.append({"label": label, "kind": kind, "file": rel, "bytes": os.path.getsize(g) if g.exists() else None,
                     "engine": engine, "humaneval": he_score(a.run, he, False),
                     "humaneval_plus": he_score(a.run, he, True),
                     # the F16 reference has no KLD row: show its own PPL (the base of every KLD row)
                     "kld": kld.get(ktag) if ktag else ({"ppl_q": ppl_f16} if rel.endswith("-F16.gguf") else None)})
    (a.out / "results.json").write_text(json.dumps({"run": str(a.run), "rows": rows}, indent=1) + "\n")

    def f(x, d=2, pct=False):
        return "—" if x is None else (f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}")

    md = ["| file | size GB | engine | HumanEval | HE+ | KLD vs F16 | same top-p % | PPL |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        k = r["kld"] or {}
        md.append(f"| {r['label']} | {f(r['bytes'] / 1e9 if r['bytes'] else None)} | {r['engine']} | "
                  f"{f(r['humaneval'], pct=True)} | {f(r['humaneval_plus'], pct=True)} | {f(k.get('kld_mean'), 4)} | "
                  f"{f(k.get('same_top_p_pct'), 1)} | {f(k.get('ppl_q'))} |")
    (a.out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
