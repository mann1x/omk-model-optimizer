# gemma-4-26B-A4B agentic sft11 — training record

The recipe is `../sft11_train.sh` with the trainer in `../omk_sft2_code` (corpus under `../../corpus`). This directory
is the run's record, exported from W&B with `tools/wandb_export_run.py` (read-only against the API):

- `wandb_bm889-g4-bf16-sft11/` — W&B run `mannix/omk-qad/48cb7c785e9619f1`, 2026-11-11, 2,108 steps, finished.
  - `run.json` — id, state, host, the exact launch command (`-m omk_qad.train --phase train --mode bf16 ... --steps 2108
    --lr 4.74e-06 ...`), git state.
  - `config.json` / `files/config.yaml` — every trainer argument as logged.
  - `history.jsonl.gz` — EVERY logged row, every key: `train/*` (loss, lr, tokens, s_per_step, peak_mem_gib,
    row_loss_min/max, loop_share, per-source losses), `opt/*` (grad_norm, update_ratio, per-class
    attn/mlp/router/experts grad_rms / update_rms / update_ratio / clip_frac / u_rms / active_frac), `eval/*` at steps
    0, 200, ..., 2000, 2108 (ppl, nll, and per eval file nll / ppl / scored: the 11 render families + the 9 sft_mix
    files + heldout rkl/klk/top1 against the teacher cache).
  - `system.jsonl.gz` — the W&B system stream (GPU utilisation / memory / power, CPU, network).
  - `files/output.log` — the trainer's console output (DATA order line, every logged step row, every `EVAL {...}` JSON line).
  - `summary.json`, `files/wandb-metadata.json`, `files/requirements.txt`, `MANIFEST.txt` (sizes + sha256).

Data order: `shuffle (seed 0)`, 4,217 rows — identical to the pruned-sft11 first try (`../../v9a_sft11_mlk8_98e`) and to
the base-pruned bm-898 run, so same-step rows of the three runs saw the same batch.

Why this exists: the run directory on bs2 (`g4_recovery/runs/g4_bf16_sft11`, with the local `.wandb` file and `log.txt`)
and the text log `train_sft11.log` were removed with the whole `g4_recovery` tree on 2026-11-17 after a small-files pull
that did not include them. The W&B API still held the complete record; this export is now the primary copy (a second
copy lives on solidpc under `backup_models/google/v9a_sft11/wandb_export/`). Rule since then: export every W&B run and
diff the small-file inventory before removing a run tree.

Re-export (needs `WANDB_API_KEY` in the environment, never on the command line):

```
python tools/wandb_export_run.py --out <dir> mannix/omk-qad/48cb7c785e9619f1
gzip -9 <dir>/omk-qad/*/history.jsonl <dir>/omk-qad/*/system.jsonl
```
