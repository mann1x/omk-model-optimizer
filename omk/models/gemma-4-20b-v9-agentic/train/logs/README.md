# gemma-4-20B-v9-agentic sft12 (render_oc33, 855 steps) — training record

Recipe: `../sft12_train.sh` (+ `../sft12_select.py`, `../sft12_export_q4km.sh`). This is the shipped model's SFT pass.

- `train_sft12_render_oc33.log`, `sft12_render_oc33.out` — the trainer text log and launcher output as kept on bs2.
- `sft12_run_log.txt` — the run directory's `stage1/log.txt`.
- `wandb_bm893-g4-98e-sft12-render_oc33/` — W&B run `mannix/omk-qad/f5502b2581256419` (2026-11-14, 855 steps)
  exported with `tools/wandb_export_run.py`: `run.json` (launch command), `config.json`, `history.jsonl.gz` (every
  row, every key), `system.jsonl.gz`, `files/output.log` (console), `summary.json`, `MANIFEST.txt`. Same layout as
  `../../../gemma-4-26b-a4b/sft11/train/logs/README.md`.
