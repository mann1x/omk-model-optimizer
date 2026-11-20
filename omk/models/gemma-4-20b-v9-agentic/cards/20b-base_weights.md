---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: image-text-to-text
datasets:
- ManniX-ITA/gemma-4-26B-A4B-agentic-sft
- ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b
- nvidia/Nemotron-SFT-Agentic-v2
- nvidia/Nemotron-SFT-OpenCode-v1
- nvidia/Nemotron-SFT-SWE-v2
- nvidia/Nemotron-SFT-Competitive-Programming-v2
- nvidia/Nemotron-Instruction-Following-Chat-v1
- nvidia/Nemotron-SFT-Instruction-Following-Chat-v2
- nvidia/Nemotron-SFT-Math-v3
- nvidia/Nemotron-Science-v1
- nvidia/Nemotron-SFT-Multilingual-v1
- interstellarninja/hermes_reasoning_tool_use
- lambda/hermes-agent-reasoning-traces
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

# Gemma 4 20B v9-agentic — the 98-expert prune of Gemma 4 26B-A4B-it, retrained with the sft11 recipe

> **Internal build `base-mlk8-noa-sft11` (2026-11-19), under evaluation — not released.** It replaces the earlier
> sft11+sft12 build of this model in this repository. The owner is not satisfied with its stability in agent use
> (loops or non-convergence under anti-repetition sampling); a further SFT round that trains the whole router block is
> planned. Numbers below are the training record, not a release claim.

**What it is.** A compact Gemma 4 for agentic coding on a single 16 GB GPU: 30 layers × **98 of 128 experts**,
20.45 B parameters, ~4 B active per token (top-8 + shared), same tokenizer, chat template, attention, norms and vision
tower as [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). This build takes the
**vendor model**, prunes it with the v9-agentic `ml_k8` competence map (no shared-expert upweight), and then runs the
**sft11 recipe once** on the pruned weights — the on-policy, loop-free agentic recovery SFT that produced
[gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic). The previous build did it the
other way round (SFT the 26B first, prune, then a short sft12); this one tests prune-first.

| | 26B-A4B-agentic | **20B v9-agentic base-mlk8-noa-sft11 (this repo)** | 20B v9-agentic sft12 (previous build) |
|---|---|---|---|
| source weights | google/gemma-4-26B-A4B-it | **google/gemma-4-26B-A4B-it, pruned first** | the 26B's sft11 weights, pruned |
| experts / layer | 128 | **98** (`ml_k8` map) | 98 (`ml_k8` map) |
| shared-FFN upweight | none | **none** | none |
| post-prune training | sft11 (on the 26B) | **sft11 recipe, once, on the pruned model**: 2,108 steps × 2 × 16,384 tokens | sft12: 855 steps, short traces |
| trained tensors | — | **attention + dense MLP + router.proj + routed experts** (295 tensors, 19.14 B) | dense MLP + router + experts |
| total / active params | 25.5 B / ~4 B | **20.45 B / ~4 B** | 20.45 B / ~4 B |
| bf16 size | 51 GB | **40.9 GB** (9 shards) | 40.9 GB |
| chat template | Google official | **Google official, unmodified** | Google official |

## Files

| Format | Repo | Notes |
|---|---|---|
| **bf16** (this repo) | [`gemma-4-20B-v9-agentic`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic) | 9 shards, `SHA256SUMS`, `omk_bf16_export.json` (the 295 keys the SFT replaced; 718 verified byte-identical to the pruned source), `expert_drop_metadata.json` (the prune), `BUILD_OK`. |
| **GGUF** | [`gemma-4-20B-v9-agentic-GGUF`](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF) | F16 + 15 **K-FAC rounded** tiers Q8_0 … IQ2_XS (7–42 % less KL than the stock `llama-quantize` tier at identical size, per-tier twin table on that card), `imatrix.dat` (AtomicChat corpus, 9,590 chunks). No QAD tier, no mmproj. |

## Recipe

Prune maps and drop tool: [omnimergekit `recipes/gemma4/v9_agentic`](https://github.com/mann1x/omnimergekit/tree/main/recipes/gemma4/v9_agentic).
SFT, export and quantization, every script as run with its log:
[omk-model-optimizer `omk/models/gemma-4-20b-v9-agentic`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-20b-v9-agentic).

### 1. The prune: `ml_k8` on the vendor 26B

The published v9-agentic drop map, unchanged: 30 of the 128 routed experts removed in every layer (900 in all), router
rows sliced to match, **no rescaling of the shared (dense) FFN**, applied with omnimergekit `expert_drop.py` to
google/gemma-4-26B-A4B-it. The map comes from per-expert, per-class contribution scores on the v9 agentic competence
map (single-turn class prompts + 43 multi-turn tool-calling sessions through the chat template), with 132 force-kept
experts (120 end-of-turn routing-lift pins + 12 loop-specific pins) and 8 multilingual rescues; the full selection
parameters are on the previous build's record in the omk repo. Tokenizer, chat template, attention, norms, embeddings
and vision tower are the vendor's, untouched. `expert_drop_metadata.json` in this repo lists the kept experts.

### 2. sft11: the corpus

The corpus of gemma-4-26B-A4B-agentic, unchanged: prompts, tools and recorded tool outputs from nine NVIDIA Nemotron
SFT sets and interstellarninja/hermes_reasoning_tool_use, every trained assistant turn **regenerated by the untouched
Gemma-4-26B-A4B-it in its own chat template** (thinking on) and kept only when a validator found no loop, no foreign
token and a tool call equivalent to the recorded one; one source regenerated by Gemma-4-31B-it (Kimi-K2.5 agent
tasks). 16,296 conversations → **4,217 windows of 16,384 tokens**, one epoch. Published as
[ManniX-ITA/gemma-4-26B-A4B-agentic-sft](https://huggingface.co/datasets/ManniX-ITA/gemma-4-26B-A4B-agentic-sft) and
[ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b](https://huggingface.co/datasets/ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b).

### 3. sft11: training

omk-model-optimizer `omk_qad.train --mode bf16 --loss ce` on one RTX PRO 6000 Blackwell, **24.2 h**
(2026-11-18 12:34 → 11-19 12:48 UTC). Scope `attention, dense_mlp, router, routed_experts`: **295 tensors, 19.139 B
parameters in bf16**; embeddings (tied head), MTP head and every 1-D scale (norms, `router.scale`,
`router.per_expert_scale`) frozen (0.739 B). Cross-entropy on the regenerated assistant tokens only, batch 2 × 16,384
tokens, **2,108 steps**, peak LR 4.74e-6 with 10 warm-up steps, eval every 200 steps on the corpus' own held-out split.

Held-out NLL over the eleven training families, token-weighted: **0.489 for the pruned, untrained model → 0.389 after
the SFT**. Training record: [`train/logs/`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-20b-v9-agentic/train/logs)
(launcher `sft11_base_mlk8noa_train.sh`, run log, W&B export).

### 4. Export

`omk_qad.train --phase export --mode bf16`: **295 tensors replaced, 718 verified byte-identical** to the pruned source
(embeddings, norms, scales, vision tower), 9 shards; `omk_bf16_export.json` lists every replaced key, `SHA256SUMS`
covers every file.

## Quantization

The GGUF tiers are not plain `llama-quantize` outputs. Each tier is made by llama.cpp b12386 with the AtomicChat
corpus imatrix (9,590 chunks, rendered through the chat template, `--parse-special`) and the Gemma-4 A4B attention
rule (attention Q8_0 from Q4_K_M upward, Q6_K below), then **re-rounded with K-FAC** (`omk_qad.kfac_round`): every MLP
tensor GPTQ-rounded against the layer's Kronecker-factored curvature with the real llama.cpp encoders, into a byte copy
of the stock file. Measured on 44 held-out agentic windows against this model's own F16: Q4_K_M KL 0.00678 →
**0.00534** (−21 %), Q6_K 0.00231 → **0.00214** (−7 %). Per-tier table and method:
[GGUF card](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF),
[`docs/kfac-rounding.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/kfac-rounding.md).

## Benchmarks

Not run on this build. The quants are gated by the KL twins on the GGUF card, not by a task score; the owner's agent
tests found the build unstable, so the canonical suite is held until the next SFT round.

## Chat template and sampling

`chat_template.jinja` is **Google's official Gemma 4 template, unmodified** (18,683 B, sha256 `ae53464bf3be2580…`);
the GGUF builds embed the same file. Pass **`--jinja`** to llama.cpp. End of turn is token **106** (`<turn|>`).

`generation_config.json` carries the vendor defaults (temperature 1.0, top_k 64, top_p 0.95). The sampler the 26B
agentic parent is deployed with (temperature 0.8, top_p 0.95, top_k 64, min_p 0, presence_penalty 0.035, no
repetition penalty) was the starting point for this build's evaluation; **no sampler is recommended for this build** —
under anti-repetition sampling it either loops or fails to converge in agent sessions, which is why it is not released.

```bash
llama-server -m gemma-4-20B-v9-agentic-Q4_K_M.gguf -c 32768 -ngl 99 --jinja -fa on \
    --chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-budget 8192 \
    --reasoning-budget-message $'\n\nI have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.\nI\'ll be more terse and concise now and maybe I need to consider a different approach.\n'
```

## Lineage

google/gemma-4-26B-A4B-it → expert drop to 98/layer with the v9-agentic `ml_k8` competence map, no shared-expert
upweight → **sft11 recipe, once** (on-policy loop-free agentic SFT, one epoch, attention + dense MLP + router.proj +
experts) → bf16 export (this repo) → F16 → AtomicChat imatrix → K-FAC rounding per tier (GGUF repo). Built with
[omk-model-optimizer](https://github.com/mann1x/omk-model-optimizer) and [omnimergekit](https://github.com/mann1x/omnimergekit).

## License

Apache-2.0, as the base model. The vision tower and everything the prune and the SFT did not touch are the vendor's.
