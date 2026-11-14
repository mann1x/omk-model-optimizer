# sft7 – sft10 (bm-889, 2026-11-07 → 11-11) — unpublished experiments

Runs between sft5 and the published sft11. None of them was released; their gates and the decision to go back to the
sft2 recipe for sft11 are in the main [README](../README.md#earlier-gates-10-runs-per-checkpoint). The launchers are
here as run (`train/`); they ran from code copies of `omk_qad` cut from `62740da07` plus edits that are all in
`omk-main` since `6080de873`.

| run | launcher | corpus render | trainer features (flag on `omk-main`) | renderer features |
|---|---|---|---|---|
| sft7 | `sft7_train.sh` | `render_32k_v2m` | `--sft-weights` per source, `--bf16-latent fp16` | served-context turns, `"score": false` |
| sft7c | `sft7c_train.sh` | `render_32k_v7cm` | as sft7 | + merged no-reasoning turns (sft7b), `--score-reasoning-cap 1000` |
| sft8 | `sft8_train.sh` | `v8/render_16k_v8m` | as sft7 | as sft7b |
| sft9 | `sft9_train.sh` | `v9/render_32k_v9m` | `--bf16-latent fp16`, lr 5e-6, no source weights | as sft7b |
| sft10 | `sft10_train.sh` | `v10/render_32k_v10m` | `--sft-weights` + `--sft-epochs 2` | as sft7 |

Parity: on CPU, `omk-main` with `--sft-weights` and `--sft-epochs` gives bit-identical weights to the sft10 code copy
after the same steps.

Not reproducible from public data: these corpora (rollout replays of agent runs plus kimi31b, cp_cf and if_rlvr
slices) were never uploaded; only the sft11 corpus is on the Hub
([ManniX-ITA/gemma-4-26B-A4B-agentic-sft](https://huggingface.co/datasets/ManniX-ITA/gemma-4-26B-A4B-agentic-sft)).
The launchers keep their bs2 paths (`/mnt/sdc/omk-model-optimizer/g4_recovery`).
