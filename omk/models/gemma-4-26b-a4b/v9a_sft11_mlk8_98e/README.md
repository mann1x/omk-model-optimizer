# sft11 recipe on the 98-expert v9-agentic ml_k8 prune (bm-893, 2026-11-14, in progress)

The sft11 agentic model ([ManniX-ITA/gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic))
pruned to 98 experts per layer with the published v9-agentic map, then trained again with the sft11 recipe, to recover
what the prune costs on agentic work (the pruned Q4_K_M "struggles to converge" in agent runs).

## Weights

- drop map: omnimergekit `recipes/gemma4/v9_agentic/maps/v9ml_ladder/ml_k8.json` (sha256 `84f2bd96…`, the published
  `gemma-4-20B-v9-agentic` map), applied with omnimergekit `scripts/build_v9_agentic_sft11_quick.sh <map> mlk8-noa 1 none`
  (commit `a9a6e06`): `expert_drop.py` on the sft11 weights, **no** `router_shared_upweight` (ALPHA `none`: the shared
  expert stays as in sft11), the official chat template, sft11's generation/preprocessor config.
- result: 30 layers × 98 experts, 20.45 B parameters; the bf16 is not published.

## Training

[`train/sft11_mlk8noa_train.sh`](train/sft11_mlk8noa_train.sh) — sft11's launcher with the model swapped, run on bs2 GPU1:
same corpus (4,217 windows of the sft11 render → 2,108 steps at bs 2, lr 4.74e-6, warmup 10, seq 16,384), same scope
(`attention,dense_mlp,router,routed_experts`), same 20 held-out files (217 windows, 1,165,554 scored positions).
Differences:

- no `--eval-set heldout` (the 31B-teacher KL set was not kept); the run evaluates on its own held-out files only.
- it ran from the archived sft11 code copy with the `eval.pt` change now in `247be006a`. From `omk-main`, use the
  sft11 command of the main README with `--bf16-latent bf16` and this model as `--model`.

W&B: project `omk-qad`, run `bm893-g4-98e-mlk8noa-sft11`. Starting point (init eval): nll 1.060 / ppl 2.885
(agentic 0.428); sft11 itself started at 0.864 on the unpruned base and ended at 0.926.
