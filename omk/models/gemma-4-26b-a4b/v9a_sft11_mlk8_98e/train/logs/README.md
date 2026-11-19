# pruned-sft11 (ml_k8, no shared α, 98e) + sft11 recipe — first try, training record

Recipe: `../sft11_mlk8noa_train.sh`. The run was stopped at step 725 (eval at 600 was the best: 11-family nll 0.408);
it led to the milder sft12 arm (`../../../gemma-4-20b-v9-agentic/train`).

- `train_sft11_mlk8noa.log`, `sft11_mlk8noa_launch.out` — the trainer text log and launcher output as kept on bs2.
- `wandb_bm893-g4-98e-mlk8noa-sft11/` — W&B run `mannix/omk-qad/d953b7c5f51b373e` (2026-11-14, 748 steps) exported
  with `tools/wandb_export_run.py`: `run.json` (launch command), `config.json`, `history.jsonl.gz` (every row, every
  key), `system.jsonl.gz`, `files/output.log` (console), `summary.json`, `MANIFEST.txt`. Same layout as
  `../../sft11/train/logs/README.md`.
