# sft11 head-to-head vs the base model (SDK 4.100.277)

The comparison behind the Results section of [`../../README.md`](../../README.md) and the HF model cards: base
`gemma-4-26B-A4B-it` vs sft11 final, task `gate-owner/manic_miner_plain` (automaticum), 20 runs each, 1,800 s wall per
run, 2026-11-12/13 on bs2 (2× RTX PRO 6000 Blackwell). Files as run (host paths included).

| path | what |
|---|---|
| [`results/h2h_report.md`](results/h2h_report.md) | the full analysis: outcome, pass curve, loop rates, every metric as median [p25–p75] with Mann–Whitney p (all runs and failed runs only), tool profile, guard kinds, failed-call classes, per-run table, basis parity |
| `results/h2h_report.json` | the same aggregates, machine-readable |
| `results/h2h_runs.json` | one row per run: verdict, wall time, loop scan, run_game history, automaticum findings and all ~200 automaticum metrics |
| `automaticum/arms/` | the two arms (`g4-26b-{base,sft11}-f16kv-sdk277-lsp-h2h`) and the parent they extend (`…-sft11-sfinal-gate`, sampler) |
| `automaticum/resolved_arm_{base,sft11}.json` | the fully resolved arm, effective sampler settings, SDK, task hashes and server provenance, read from a recorded run (`run show -json`), so the chain of `extends` is not needed |
| `automaticum/endpoints/` | the two llama.cpp endpoints (capacity 1 = one run at a time) |
| `serve/start-824{4,5}-…-f16kv.sh` | the two server launchers: identical except model and GPU (opencoti llama.cpp, drafter `gemma-4-26B-A4B-it-assistant` Q8_0, `--spec-draft-n-max 2`, f16 KV, sampler defaults = the arm's) |
| `watch_runs.sh`, `sft11_run_summary.{sh,py}`, `loopscan.py`, `timeline.py` | the live per-run watcher: saves each finished run's events stream, runs the loop scan and timeline, prints one line per run |
| `h2h_report.py` | the aggregator (runs on the automaticum host, stdlib only): Fisher exact, Wilson intervals, Mann–Whitney U |
| `h2h_card_section.py`, `h2h_cards_build.py` | render the HF card sections from `h2h_report.json` (no number typed by hand) |

## Reproduce

```bash
# on the automaticum host, with both servers up (serve/) and the arms/endpoints installed
automaticum run start -arm g4-26b-base-f16kv-sdk277-lsp-h2h  -task gate-owner/manic_miner_plain -n 20 -wall-s 1800
automaticum run start -arm g4-26b-sft11-f16kv-sdk277-lsp-h2h -task gate-owner/manic_miner_plain -n 20 -wall-s 1800
# per finished run (the watcher does this): save events, loop scan, timeline
bash sft11_run_summary.sh <run-id> <verdict> <wall_ms> h2h277sft
# aggregate once both batches are 20/20 (add --partial for an interim report)
python3 h2h_report.py <out-dir> \
    base:g4-26b-base-f16kv-sdk277-lsp-h2h:<base-batch-prefix>:<live-dir-base> \
    sft11:g4-26b-sft11-f16kv-sdk277-lsp-h2h:<sft-batch-prefix>:<live-dir-sft11>
```

Batches used: base `20261112-220607-9fb9ab`, sft11 `20261112-220607-0cbea6`.

## Reading the numbers

- **Pass = finished within the 30-minute budget**, not an absolute success rate.
- **Line-90 fields** (in `h2h_runs.json`) follow the broken line by line number. They are unreliable once a run
  restores the file or edits it with search/replace, which both models do, so the report does not tabulate them. The
  oracle verdict is the ground truth.
- **Server speed** differs by 4 % between the two servers. Draft acceptance, summed from the server logs over the
  head-to-head, was 89.75 % (base) vs 88.70 % (sft11).
