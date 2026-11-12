# Mode 3 — Recovery training to BF16 (`--mode bf16`)

**What it does.** Trains a model's **own bf16 weights**: no quantizer, no layout, no GGUF types. The output is a
Hugging Face checkpoint in the source's own key names and shard layout, with only the trained tensors replaced
(every other tensor, the vision tower included, verified byte for byte). Quantize it afterwards like any HF model
(GGUF via llama.cpp / omnimergekit `quantize_gguf.py`, or [QAD](qad.md) on top of it).

Two ways to use it:

| | loss | targets | use it for |
|---|---|---|---|
| **distillation recovery** | `--loss jaccard \| kl \| rkl` | a teacher's cached top-K (any model of the same tokenizer, a bigger sibling included) | repair a damaged model (merges, expert drops, earlier fine-tunes) against a stable teacher; per-role targets for agentic traces |
| **SFT recovery** | `--loss ce` | the scored tokens of role-masked rendered conversations (`--sft-data`) | teach behaviour from curated traces (the Gemma-4 26B-A4B agentic model) |

Memory: the trained tensors are latents updated inside backward by the momentum-free Adafactor QAD uses, written
back to bf16 with stochastic rounding. There is **no fp32 master copy**, so a 26B MoE trains on one 96 GB GPU:
Gemma-4 26B-A4B, all of attention + dense MLP + router + routed experts (295 tensors, 24.5 B parameters), seq 16,384,
bs 2: **74.3 GiB** peak on an RTX PRO 6000 Blackwell, ~37 s/step.

## Training scope (`--train-scope`)

| scope | trains |
|---|---|
| `gemma4` (default) | the reviewed Gemma-4 MoE scope: every `self_attn.{q,k,v,o}_proj` + the per-layer dense `mlp.{gate,up,down}_proj`. Count derived from the model (Gemma-4 full-attention layers share K/V and have no `v_proj`); refuses if anything outside the regex matches. `--train-router` adds `router.proj`. |
| `classes:<a,b,...>` | census classes: `attention`, `dense_mlp`, `router`, `routed_experts`, `shared_experts`, `linear_attn`, ... e.g. `classes:attention,dense_mlp,router,routed_experts` (the agentic run) |

The embedding / head, norms and vision tower are never trained. Final-logit soft capping (Gemma-4: 30) is applied in
the teacher cache, the eval and the loss. `--bf16-latent fp16|bf16` sets how trained weights are stored between steps.

## 1. Data: conversations → role windows (`omk_qad.rollouts`)

Input rows are OpenAI chat JSONL, `{"id", "messages", "tools"?}` (assistant `reasoning_content`, `content`,
`tool_calls`; `tool` results). `rollouts` renders them with **the student's own chat template** and gives every
token a role:

| role | positions | trained as |
|---|---|---|
| 0 ignore | prompt, system, user, tool output, template text the model did not generate | not scored |
| 1 teacher | generated reasoning, prose, tool-call structure, the stop token | teacher KL (distillation) / CE (SFT) |
| 2 anchor | generated code (code-carrying tool arguments, fenced code) | KL to the original student (`--anchor` cache pass) |
| 3 loop / 4 lead | text inside / just before a marked loop span | teacher KL + unlikelihood (`--ul-weight`, `--loop-weight`) |

```bash
ENV=$(conda info --base)/envs/omk-modelopt/bin/python
CUDA_VISIBLE_DEVICES= $ENV -m omk_qad.rollouts --model STUDENT_DIR --in conv.train.jsonl \
    --history-reasoning strip --seq 16384 --pack --out render/src.train.roles.jsonl
```

- `--history-reasoning` must match how the serving client re-sends earlier reasoning: `keep`, `strip` (none, the
  Gemma-4 template default) or `last-inline` (Cerebriline SDK ≥ 4.100.260). `strip` / `last-inline` give one example
  per assistant turn.
- Gemma-4 drops the reasoning of turns before the last user message, so conversations are split into segments at
  later user messages; every scored turn's prompt must be an exact prefix of its render, or the row is refused.
- `--pack` puts short examples back to back in one window; long segments become stride-overlapping windows that
  score each token once. Each output has a `.stats.json` (rows, windows, refusals, `"complete": true`, sha256)
  that the trainer checks.
- `--capture DIR` builds from the Cerebriline SDK's "Save model requests" files instead (`--verify-server URL`
  checks every prompt against llama-server's `/apply-template`).
- Render a **held-out** split of each source the same way (`*.eval.roles.jsonl`): `--loss ce` refuses to run
  without `--sft-eval`, and evaluates CE per source on those rows.

## 2. Cache

Distillation needs the teacher's top-K on every training window:

The commands that built the Gemma-4 cache `cache_teacher_r4` (teacher = Gemma-4-31B-it BF16 GGUF, student =
Gemma-4-26B-A4B-it):

```bash
M=gemma-4-26B-A4B-it; T=gemma-4-31B-it-BF16.gguf; TB=omk_qad/llama_topk/llama_topk
# teacher top-K over the training windows; with --teacher-gguf, --model gives the tokenizer only (no imatrix)
$ENV -m omk_qad.train --phase cache --mode bf16 --model $M --teacher-gguf $T --teacher-gguf-bin $TB \
    --cache cache_t --data "pool.roles.jsonl=2849,replay.roles.jsonl=521" --n-seq 3370 --seq 16384 \
    --imatrix-seqs 0 --eval-data heldout_eval.jsonl --n-eval 8
# held-out teacher-KL eval set beside the cache (rollouts the cache never trained on)
$ENV -m omk_qad.train --phase evalset --mode bf16 --model $M --teacher-gguf $T --teacher-gguf-bin $TB \
    --cache cache_t --seq 16384 --topk 64 --eval-set heldout --eval-data heldout.roles.jsonl --n-eval 27
# anchor targets (role 2): the ORIGINAL student's top-K on the same windows
$ENV -m omk_qad.train --phase cache --mode bf16 --anchor --model $M --cache cache_t --seq 16384
```

A teacher that fits in torch is passed as `--model TEACHER_DIR` without `--teacher-gguf`.

`llama_topk` build line: `omk_qad/gguf_teacher.py`. Only role ≥ 1 positions are computed; llama.cpp applies
Gemma-4's soft cap in the graph.

**SFT (`--loss ce`) takes no targets from the cache.** The training rows and their order come only from `--sft-data`
and `--seed`. The cache provides `cache.json`, its eval rows and the optional `--eval-set` sets, which are
diagnostics (teacher-KL on held-out agentic windows) and never change the weights.

## 3. Train

```bash
# SFT recovery (the agentic recipe's flags)
$ENV -m omk_qad.train --phase train --mode bf16 --model STUDENT_DIR --cache cache_t --out runs/r \
    --train-scope classes:attention,dense_mlp,router,routed_experts --loss ce --mtp freeze \
    --sft-data "$(ls render/*.train.roles.jsonl | paste -sd,)" --sft-eval "$(ls render/*.eval.roles.jsonl | paste -sd,)" \
    --seq 16384 --bs 2 --steps 2108 --lr 4.74e-6 --fp-lr 4.74e-6 --warmup 10 \
    --eval-set heldout --eval-every 200 --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3

# distillation recovery (teacher + anchor + loop roles; the Gemma-4 run4b flags)
$ENV -m omk_qad.train --phase train --mode bf16 --model STUDENT_DIR --cache cache_t --out runs/d \
    --train-scope classes:attention,dense_mlp,router,routed_experts --loss rkl \
    --anchor-weight 1.0 --ul-weight 1.0 --loop-weight 1.0 --bs 2 --steps 1685 --lr 1e-5 --warmup 10 \
    --eval-set heldout
```

- `--loss ce` requires `--mtp freeze` and `--sft-eval`; the run needs `steps × bs × accum` rows (one epoch =
  windows / bs). Every eval logs CE per source and, with `--eval-set`, teacher KL / top-1 on the held-out set.
- The lossless gate (unquantized student vs teacher) runs only when the cache teacher *is* the student.
- Snapshots: keep numbered `ckpt_stepNNNNNNN.pt` (`--snapshot-every`, `--snapshot-keep`) so a mid-run step can be
  exported and gated without stopping the run. Pause / resume / divergence guard work as in [QAD](qad.md#3-train).

## 4. Export to an HF checkpoint

```bash
CUDA_VISIBLE_DEVICES= $ENV -m omk_qad.train --phase export --mode bf16 --model STUDENT_DIR \
    --load runs/r/stage1/final.pt --out-hf out/model-recovered        # or a ckpt_stepNNNNNNN.pt snapshot
```

The export writes `omk_bf16_export.json` (trained keys, `replaced` / `unchanged_verified` counts, step, checkpoint)
next to the shards. The new dir must not exist. Then build GGUFs from it the usual way; for Gemma-4 keep the official
chat template and `eos = 106` in the GGUF (see the model section).

Worked example, end to end and reproducible: [Gemma-4 26B-A4B agentic](../models/gemma-4-26b-a4b/README.md).
