# gemma-4-20B-v9-agentic sft12 (render_oc33, 855 steps) — training record

Recipe: `../sft12_train.sh` (+ `../sft12_select.py`, `../sft12_export_q4km.sh`). This is the shipped model's SFT pass.

- `train_sft12_render_oc33.log`, `sft12_render_oc33.out` — the trainer text log and launcher output as kept on bs2.
- `sft12_run_log.txt` — the run directory's `stage1/log.txt`.
- `wandb_bm893-g4-98e-sft12-render_oc33/` — W&B run `mannix/omk-qad/f5502b2581256419` (2026-11-14, 855 steps)
  exported with `tools/wandb_export_run.py`: `run.json` (launch command), `config.json`, `history.jsonl.gz` (every
  row, every key), `system.jsonl.gz`, `files/output.log` (console), `summary.json`, `MANIFEST.txt`. Same layout as
  `../../../gemma-4-26b-a4b/sft11/train/logs/README.md`.

# gemma-4-20B-v9-agentic-base-mlk8-noa sft11 (bm-898, 2108 steps) — training record

The alternative order (prune the 26B vendor base with ml_k8 / no shared α to 98e, then the sft11 recipe ONCE on it)
vs the shipped sft11 → prune → sft12. Plan: `backup_models/docs/plans/plan_v9_agentic_base_mlk8noa_sft11.md`.

- `train_sft11_base_mlk8noa.log` — the trainer text log as kept on bs2 (SFTB_EVAL rows every 200 steps, SFTB_OK
  2026-11-19T12:48:26Z); `sft11_base_mlk8noa_run_log.txt` — the run directory's `stage1/log.txt`.
- `wandb_bm898-g4-98e-base-mlk8noa-sft11/` — W&B run `mannix/omk-sft-20b-base-mlk8noa/5cb8776c0afeca1d`
  (2026-11-18/19, 2108 steps, 2109 history rows, state finished) exported with `tools/wandb_export_run.py`, same
  layout as above. Final eval (step 2108): 11-family token-weighted nll 0.3894 (0.489 at step 0), code-4 0.257,
  chat-4 0.435.
