---
license: apache-2.0
base_model:
- ManniX-ITA/gemma-4-26B-A4B-agentic
- google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
- multilingual
tags:
- agentic
- code
- expert-pruning
- gemma4
- moe
- omnimergekit
- omk-model-optimizer
- reasoning
- sft
- surgery
- tool-calling
model_name: gemma-4-20B-v9-agentic
author: ManniX-ITA
architecture: Gemma4
parameters: 20.45B
active_parameters: 4B
format: safetensors
features:
- MoE
- agentic
- code
- expert-pruning
- multilingual
- reasoning
- tool-calling
- vision
---

# Gemma 4 20B v9-agentic — the 98-expert prune of the loop-free agentic 26B, retrained (sft12)

> **Build sft12 (2026-11-16).** This repository holds the **sft12** weights: the v9-agentic 98-expert prune applied to
> [gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic) (not to the stock `-it`),
> **without the shared-expert α=1.2 upweight** of the earlier builds, then one round of short-trace agentic SFT.
> The GGUF repo is rebuilt from it with K-FAC rounding. The 2026-10-17 build this replaces is described at the end,
> with its benchmark numbers, for reference only.

**What it is.** A compact Gemma 4 for **agentic coding on a single 16 GB GPU**: 30 layers × **98 of 128 experts**,
20.45 B parameters, ~4 B active per token (top-8 + shared), same tokenizer, chat template, attention and norms as the
26B. The parent is the on-policy loop-free SFT of Gemma 4 26B-A4B-it; the prune keeps the experts that carry
tool-calling, code and end-of-turn routing; sft12 trains the pruned model's MLPs back toward the parent on its own
agentic traces, short ones only.

| | 26B-A4B-agentic (parent) | **20B v9-agentic sft12** | 20B v9-agentic 2026-10 (previous) |
|---|---|---|---|
| source weights | google/gemma-4-26B-A4B-it + sft11 | **the parent's sft11 weights** | google/gemma-4-26B-A4B-it |
| experts / layer | 128 | **98** (`ml_k8` map) | 98 (`ml_k8` map) |
| shared-FFN upweight | none | **none** | α = 1.2 on `mlp.down_proj` |
| post-prune training | — | **sft12**: 855 steps, agentic/hermes/kimi/opencode + IFEval anchor, attention frozen | none |
| total / active params | 25.5 B / ~4 B | **20.45 B / ~4 B** | 20.45 B / ~4 B |
| bf16 size | 51 GB | **40.9 GB** (9 shards) | 40.9 GB |
| chat template | Google official | **Google official, unmodified** | Google official |
| GGUF | K-FAC ladder | **K-FAC ladder + QAD IQ2_XXS** | llama-quantize ladder |

## Quantized formats

| Format | Repo | Notes |
|---|---|---|
| **bf16** (this repo) | [`gemma-4-20B-v9-agentic`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic) | 9 shards, 40.90 GB, `SHA256SUMS`, `omk_bf16_export.json` (what the SFT changed). |
| **GGUF** | [`gemma-4-20B-v9-agentic-GGUF`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF) | **K-FAC rounded** tiers Q8_0 … IQ2_XS at identical size to the stock `llama-quantize` tier with 6–24 % less KL, **IQ2_XXS from a seeded QAD**, F16, `imatrix.dat` (AtomicChat corpus, 9,590 chunks), `mmproj`. Per-tier KL twin table on that card. |
| **Ollama** | `mannix/gemma-4-20B-v9-agentic` | The public tags are the previous build until re-pushed; the sft12 K-FAC tags are published separately. |

## Recipe

Everything is in two public repos: the prune maps and the drop tool in
[omnimergekit `recipes/gemma4/v9_agentic`](https://github.com/mann1x/omnimergekit/tree/main/recipes/gemma4/v9_agentic),
the SFT, export, quantization and QAD in
[omk-model-optimizer `omk/models/gemma-4-20b-v9-agentic`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-20b-v9-agentic)
(every script as run, with the numbers).

### 1. The prune: the `ml_k8` map on the agentic 26B

The drop map is the published v9-agentic map, unchanged (`v9ml_ladder/ml_k8.json`, sha256 `84f2bd96…`), applied with
omnimergekit `expert_drop.py` to the **sft11 final weights of gemma-4-26B-A4B-agentic**. The map was produced by
`generate_drop_map_v5fk.py` from per-expert, per-class contribution scores on the **v9 agentic competence map**:

- **calibration**: Tier A — 2,048-token single-turn prompts per class (generic math / logic / code / science /
  creative and one class per software-engineering language); Tier B — **43 multi-turn tool-calling sessions with
  reasoning**, scored through the chat template.
- **selection**: target 98 (30 dropped per layer), `protect_top 16`, `alpha 2.0`, `strategy max`, `normalize rank`,
  `breadth_bonus 0.5`, per-layer floor map (15–25, mean 16.9, ranked on the v7 base map), `outlier_mode median`,
  tie-break baseline `teacher_force_98e_p16_clean.json`.
- **class weights**: `generic_code` 3×, the seven `targeted_swe_*` language classes (Java, Rust, JavaScript, Ruby, Go,
  PHP, C/C++) 2× each, everything else 1×.
- **force-keep (132 experts)**: 120 end-of-turn lift pins (top-4 experts per layer by routing lift on `<turn|>`,
  id 106, over 71 agentic sessions) + 12 loop-specific pins (loop mass ÷ clean mass, `0:62 1:114 3:68 21:111 8:0 0:100
  7:63 19:116 8:85 18:102 25:122 5:85`).
- **multilingual / IFEval retargeting**: 8 experts the map drops that rank highest on a `generic_multilingual` class
  (95 prompts) are rescued, each evicting the lowest-scoring survivor of its layer (`rescued 0:9 8:122 15:64 15:106
  23:116 26:16 27:48 29:25`, `evicted 0:23 8:96 15:3 15:38 23:48 26:108 27:106 29:26`). Hold-out: IFEval rows 0–99
  are never in calibration.

**No shared-expert upweight.** Every earlier coder/agentic prune in this line multiplied the always-on shared FFN's
`mlp.down_proj` by α = 1.2 after the drop. The 2026-11-13/16 A/B on the owner's own agent sessions (xollama, Q4_K_M,
map variants refA / A_sft11 / hub_k12 / ml_k8, α 1.2 vs none) found the **no-upweight builds converge better**, and a
last test applying α 1.2 to the finished sft12 did not work; the shared expert is therefore left exactly as in the
parent. Build: omnimergekit `scripts/build_v9_agentic_sft11_quick.sh <map> mlk8-noa 1 none` (`ALPHA=none`).

### 2. sft12: the corpus

Selected from the parent's rendered sft11 training windows (16,384 tokens, one assistant turn scored per segment;
the parent's own loop-free regenerated turns) by `sft12_select.py`:

- **kept**: `agentic`, `opencode` (1 in 3 segments), `hermes`, `kimi31b` — the agentic map's job — plus an
  **IFEval anchor** (`ifchat_v1` + `ifchat_v2`) sized to 10 % of the estimated loss.
- **dropped**: `multilingual`, `science`, `cp`, `math`, `swe` — long turns and capacity the prune removed (the prune
  costs 0.15–0.41 nats on those held-out families and ~0 on code; a 98-expert model cannot recover them).
- **no long traces**: any segment with more than 1,024 scored tokens is dropped, not truncated (a cut turn teaches a
  turn that never ends); every kept segment ends on `<turn|>` or `<|tool_response>`.
- result: **1,710 windows → 855 steps at batch 2**, 3.06 M scored tokens; estimated loss share agentic 64 %, hermes
  15 %, IFEval 10 %, kimi31b 6 %, opencode 5 %.

### 3. sft12: training

omk-model-optimizer `omk_qad.train --mode bf16 --loss ce` on one RTX PRO 6000 Blackwell (bs2 GPU1, 10.7 h):
**scope `dense_mlp, router, routed_experts` — attention frozen**; batch 2 × 16,384 tokens, **lr 2e-6** (the parent's
sft11 used 4.74e-6), warmup 50, cosine, one epoch (855 steps), MTP head frozen, divergence guard 3×, eval and
checkpoint every 100 steps. A **stop rule** paused the run when any code family's held-out nll rose above its start
(opencode / swe / cp +0.01, kimi31b +0.02): it fired once at step 100 on 8 kimi31b rows (eval noise, +0.002 at step
200) and the run resumed from that checkpoint.

Held-out nll on the parent's 20 evaluation files (217 windows, 1,165,554 scored positions), start → step 855:

| | overall | agentic | opencode | kimi31b | swe | cp | hermes | ifchat_v1 | ifchat_v2 | multilingual | math | science |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pruned, before sft12 | 1.060 | 0.428 | 0.222 | 0.377 | 0.294 | 0.231 | 0.243 | 0.721 | 0.657 | 0.529 | 0.219 | 0.423 |
| **sft12 (this model)** | **0.977** | **0.392** | 0.225 | 0.379 | 0.296 | 0.231 | **0.232** | **0.619** | **0.554** | **0.468** | 0.219 | **0.385** |
| parent 26B (sft11 final) | 0.926 | 0.303 | 0.212 | — | 0.286 | — | — | 0.339 | — | — | — | — |

Perplexity 2.885 → 2.658. The code families stay within +0.003 nats of their start (the point of the stop rule); the
agentic, hermes and instruction-following families recover a third to a half of what the prune cost.

### 4. Export

`omk_qad.train --phase export --mode bf16`: **180 tensors replaced** (30 layers × routed experts gate_up + down, dense
gate / up / down, router), **833 verified byte-identical** to the pruned source (attention, norms, embedding, vision
tower), 9 shards — `omk_bf16_export.json` in this repo lists every replaced key. `SHA256SUMS` covers every file.

## Quantization

The GGUF tiers are not plain `llama-quantize` outputs. Each tier is first made by llama.cpp b12386 with the
AtomicChat corpus imatrix (9,590 chunks, rendered through the chat template, `--parse-special`) and the Gemma-4 A4B
attention rule (attention Q8_0 from Q4_K_M upward, Q6_K below), then **re-rounded with K-FAC**
(`omk_qad.kfac_round`): every MLP tensor is [GPTQ](https://arxiv.org/abs/2210.17323)-rounded against the layer's [Kronecker-factored curvature](https://arxiv.org/abs/1503.05671) with the
real llama.cpp encoders, into a byte copy of the stock file — same types, same size, same metadata, less error.
Measured on 44 held-out agentic windows against this model's own F16: Q4_K_M KL 0.00981 → **0.00822** (−16 %),
Q6_K 0.00311 → **0.00292** (−6 %). The IQ2_XXS tier is a quantization-aware-distillation student seeded from the
K-FAC rounding of its layout. Method and per-tier table:
[GGUF card](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF),
[`docs/kfac-rounding.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/kfac-rounding.md).

## Benchmarks

The canonical 9-bench suite runs on the **K-FAC Q6_K** (llama.cpp b12386, two RTX 3090, greedy with
`presence_penalty 0.025`, scores from `summary.json`); the table with every cell is on the
[GGUF card](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF#benchmarks) and is refreshed as benches
finish. So far: **HumanEval 98.17**, **HumanEval+ 90.85** (previous build, different sampler: 98.17 / 90.24).

## Chat template

`chat_template.jinja` is **Google's official Gemma 4 template, unmodified** (18,683 B, sha256 `ae53464bf3be2580…`);
the GGUF builds embed the same file. Pass **`--jinja`** to llama.cpp.

## Recommended sampling

```
temperature      0.8
top_k            64
top_p            0.95
min_p            0
presence_penalty 0.035   (0.025 until 2026-11-16; raised after testing)
repeat_penalty   1.0   (none)
```

The same setting as the parent's head-to-head test. The previous build needed `repeat_penalty 1.06 / repeat_last_n 96`
to stay out of loops; this one does not — the parent's SFT and sft12 carry the loop protection, and a repetition
penalty costs an agent part of its ability to converge. `generation_config.json` carries the subset `transformers`
reads.

```bash
llama-server -m gemma-4-20B-v9-agentic-Q4_K_M.gguf -c 32768 -ngl 99 --jinja -fa on \
    --chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-budget 8192 \
    --reasoning-budget-message $'\n\nI have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.\nI\'ll be more terse and concise now and maybe I need to consider a different approach.\n' \
    --temp 0.8 --top-k 64 --top-p 0.95 --min-p 0 --presence-penalty 0.035
```

## Reasoning budget and thinking stop phrase (llama.cpp)

On a hard prompt this model family can reason until it has consumed the whole context window. llama.cpp bounds the
thinking block and tells the model *why* the block is being closed:

| flag | meaning |
|---|---|
| `--reasoning-budget N` | `-1` unrestricted (default), `0` close the block immediately, `N > 0` cap it at N tokens |
| `--reasoning-budget-message` | text written into the block just before the closing tag is forced |
| `--jinja` | required — the delimiters come from the chat template (`<|channel>thought` … `<channel|>`) |

Needs llama.cpp **b8508** or newer for the flags, **b10091** or newer for the per-request overrides
(`thinking_budget_tokens`, `reasoning_budget_message`). Keep `-c` several times larger than the budget; the budget is
per thinking block, so a multi-turn agent gets a fresh window each turn. The stop phrase above is written for tool use.
xOllama tags ship it as `think_budget high` + `think_budget_message`.

## Intended use

Agentic coding on a 16 GB GPU (13.8 GB at Q4_K_M): tool-calling loops, multi-step edits across a repository, code
reasoning, prompts and replies in languages other than English. Serve with the reasoning parser enabled
(`--reasoning-parser gemma4` on vLLM; `--jinja --reasoning-budget 8192` on llama-server).

## Limitations

A research prune, not an official Google release. Expert pruning trades breadth for size: the long-form chat, science
and multilingual families lose 0.15–0.41 nats against the parent and sft12 recovers only part of it (it does not train
on them). Benchmarks of this build are being completed; until then, only the KL twins and the two code cells above are
measured on it. Below about 3 bits the Gemma 4 MoE degrades even with the trained IQ2_XXS; prefer Q4_K_M or higher.

## Lineage

google/gemma-4-26B-A4B-it → [gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic)
(sft11, on-policy loop-free SFT) → `ml_k8` expert drop to 98/layer, no shared upweight → **sft12** → K-FAC GGUF ladder
+ QAD IQ2_XXS. Predecessors in the pruned line: v7-coder → v9-agentic (2026-10, from the stock `-it`).

<details>
<summary><b>Previous build (2026-10-17): recipe differences and its benchmark numbers</b></summary>

The 2026-10-17 `gemma-4-20B-v9-agentic` was the same `ml_k8` map applied to the **stock** `google/gemma-4-26B-A4B-it`
with the mandatory **shared-FFN α = 1.2** upweight (`router_shared_upweight.py --alpha 1.2 --target mlp.down_proj.weight`)
and **no post-prune training**. Its numbers, all on **its own Q6_K**, llama.cpp b9700, greedy with
`repeat_penalty 1.06 / repeat_last_n 96 / presence_penalty 0.025`, are a different model's and are kept here only
because the name is the same:

| Benchmark | 128e (unpruned) | v7-coder | v9-agentic 2026-10 |
|---|---:|---:|---:|
| GPQA-diamond (198q) | 67.17 | 51.52 | 55.56 |
| AIME (30q) | 73.33 | 80.00 | 86.67 |
| MATH500 (100q) | 92.00 | 95.00 | 81.00 |
| GSM8K (100q) | 89.00 | 91.00 | 97.00 |
| ARC-Challenge (full) | 96.50 | 92.15 | 86.77 |
| IFEval (100q, strict) | 97.00 | 92.00 | 96.00 |
| HumanEval (164) | 97.56 | 98.17 | 98.17 |
| HumanEval+ (164) | 92.07 | 92.07 | 90.24 |
| LCB-medium-55 v4 | 96.36 | 98.18 | 98.18 |
| LCB-medium-100 v4 | 97.00 | 94.00 | 97.00 |
| LCB-hard-77 | 79.22 | 84.42 | 93.51 |
| MultiPL-E (100 × 3) | 90.00 | 89.67 | 90.00 |

GPQA by domain (previous build): physics 83.72, chemistry 33.33, biology 36.84. Repetition loops across its
benchmark answers: 3 (128e 10, v7-coder 16, different samplers). Tool calling, `tool-eval-bench` v2.6.0 hardmode
(88 scenarios, 176 pts, Q4_K_M, 64k context, 5 seeds): **137.6 ± 5.4**, 7th of 11 in that cohort; weakest axis
safety (prompt-injection and sleeper-injection scenarios failed on every seed — treat tool output as untrusted input
when you deploy any model of this line). The previous card's full text, with the per-cell sample checks, is in the
repository history before the 2026-11-16 squash.

</details>
