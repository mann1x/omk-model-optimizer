---
license: apache-2.0
base_model:
- ManniX-ITA/gemma-4-20B-v9-agentic
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
- multilingual
tags:
- gguf
- imatrix
- kfac
- quantized
- agentic
- tool-calling
- gemma4
- moe
- reasoning
- omk-model-optimizer
model_name: gemma-4-20B-v9-agentic-GGUF
author: ManniX-ITA
architecture: Gemma4
parameters: 20.45B
active_parameters: 4B
format: GGUF
quantizations:
- F16
- Q8_0
- Q6_K
- Q5_K_M
- Q5_K_S
- Q4_K_M
- Q4_K_S
- IQ4_XS
- IQ4_NL
- Q3_K_M
- IQ3_M
- IQ3_XS
- IQ3_XXS
- IQ2_M
- IQ2_S
- IQ2_XS
- IQ2_XXS
features:
- MoE
- agentic
- code
- expert-pruning
- imatrix
- kfac
- reasoning
- tool-calling
- vision
---

# gemma-4-20B-v9-agentic-GGUF — K-FAC quantizations of the sft12 build

GGUF quantizations of [ManniX-ITA/gemma-4-20B-v9-agentic](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic),
the 98-expert prune of [gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic) (the
loop-free recovery of Gemma 4 26B-A4B-it) retrained on short agentic traces — 30 layers × 98 experts, 20.45 B
parameters, ~4 B active. Recipe, training and evaluation are on the
[weights card](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic).

> **Build replacement (2026-11-16).** This repo now holds the **sft12** model. Every tier in the table below was made
> from the sft12 F16 with the recipe in the next section and **replaces** the same-named file of the 2026-10-17 build.
> Files of the previous build that have no replacement yet are listed under *Previous build* at the end; they are a
> different model (the stock `-it` pruned with the α=1.2 shared-FFN upweight).

**Every tier here is K-FAC rounded**, not a plain `llama-quantize` output: same file format, same layout, same size as
the stock tier, but the MLP weights are re-rounded against the model's own curvature, which removes 6–24 % of the
quantization error at identical size (measured below, tier by tier). The 2-bit tier (IQ2_XXS) goes one step further:
it is a trained student (quantization-aware distillation) that starts from the K-FAC rounding.

## How the quants are made

Toolchain: [omk-model-optimizer](https://github.com/mann1x/omk-model-optimizer)
([`docs/kfac-rounding.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/kfac-rounding.md),
[model folder with every script as run](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-20b-v9-agentic))
and [omnimergekit](https://github.com/mann1x/omnimergekit) `scripts/quantize_gguf.py`; llama.cpp **b12386** for every
step.

1. **F16** — `gemma-4-20B-v9-agentic-F16.gguf`, converted from the sft12 safetensors (658 tensors, `eos = 106`
   `<turn|>`, the official Gemma 4 chat template embedded, 18,683 B, sha256 `ae53464b…`). No MTP head.
2. **imatrix** — [`imatrix.dat`](./imatrix.dat) (sha256 `3b23762e…`): the
   [AtomicChat](https://huggingface.co/AtomicChat) calib-corpora recipe `gemma-4-20b-v9-agentic` **rendered through the
   Gemma 4 chat template** (3,043 documents, 4.91 M tokens, real `<|turn>` / `<|channel>` / `<|tool_call>` structure),
   **all 9,590 chunks, `--parse-special`**, computed on the sft12 F16. Every tier carries its `quantize.imatrix.*`
   keys (file, 9,590 chunks, 295 entries).
3. **stock tier** — `llama-quantize --imatrix imatrix.dat` at the tier, with the **Gemma-4 A4B attention rule**:
   the attention projections (`attn_q/k/v/output`) are **Q8_0 from Q4_K_M upward and Q6_K below** (they are small and
   the MoE is sensitive to them). This file is the reference: it is what you would get from llama.cpp alone, and it is
   scored in the table as *llama-quantize + imatrix*.
4. **K-FAC rounding** — `omk_qad.kfac_round --stems all --qtype auto --ref <stock tier>`: every MLP tensor (routed
   experts' gate_up and down, dense gate/up/down) is re-rounded by GPTQ at the encoder's own 256-column block, with
   the real llama.cpp encoders (byte-identical codes) and the layer's Kronecker-factored curvature `H ≈ G ⊗ A`
   (`A` = input second moment, `G` = sampled-label Fisher of the output; 512 windows × 2,048 tokens, attention-sink
   positions excluded). Output = a byte copy of the stock file with the MLP payloads replaced, so the two differ only
   by the rounding: same tensor types, same bytes count, same metadata.
5. **twin score** — both files are scored against the model's own F16 (llama.cpp kernels on both sides) on 44
   held-out agentic windows, 256,853 assistant positions: mean top-256 KL(F16 ‖ quant) and top-1 agreement. The K-FAC
   file is published only when it beats its stock twin; the stock twin is not published.

## Files

<!-- OMK:TIERS -->

**Reading the table.** `KL` = mean KL divergence of the quant's next-token distribution from the F16's over the
F16's top 256 tokens, on the held-out assistant positions (lower is better; 0 = identical); `top-1` = share of
positions where both pick the same token. `ΔKL` = the K-FAC file's KL relative to its stock twin at identical size.
`bpw` = 8 × bytes ÷ the GGUF's tensor element count. sha256 is the file's; the full value is in each `.gguf.sha256`.

**Other files.** `mmproj-gemma-4-20B-v9-agentic-F16.gguf` — the Gemma 4 SigLIP vision tower for multimodal use; it is
untouched by the prune and by the SFT (vision was not trained), so the projector of the first build stays valid.
`imatrix.dat` — the importance matrix above, for audit and reproduction.

### IQ2_XXS — the trained 2-bit tier

At 2 bits a rounding is not enough; the IQ2_XXS file is a **quantization-aware distillation** student
([`docs/qad.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/qad.md)): routed experts'
gate_up at IQ2_XXS, routed down at IQ4_NL, dense MLP Q4_K / IQ4_NL, attention Q6_K, embedding Q5_K. It starts from
the **K-FAC rounding of exactly that layout** (`--init-latents`, checked by a seed gate to 0.0005 KL), then trains the
fp16 latents of every quantized tensor against the F16 teacher's top-64 logits for 600 steps (seq 4,096, lr 2e-6 →
1e-7 cosine, KL loss, the quantizer in the loop at every step) on the model's own agentic corpus, and exports the
latents with the same encoders — the file runs on stock llama.cpp. Its imatrix is the run's own corpus imatrix
(16,384-token windows, assistant positions); its row in the table shows the trained student's score against the
stock IQ2_XXS twin. On the 26B of the same family the seeded QAD removes ~35 % of the stock tier's KL (rounding 17 %,
training the rest).

## How to use

```bash
llama-server -m gemma-4-20B-v9-agentic-Q4_K_M.gguf -c 32768 -ngl 99 --jinja -fa on \
    --chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-budget 8192 \
    --reasoning-budget-message $'\n\nI have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.\nI\'ll be more terse and concise now and maybe I need to consider a different approach.\n' \
    --temp 0.8 --top-k 64 --top-p 0.95 --min-p 0 --presence-penalty 0.035
```

- **`--jinja` is required**: it uses the embedded official template and gives the reasoning budget its delimiters.
- **Sampler** (the one the sft12 model is tested and deployed with, the same as the 26B agentic): temperature 0.8,
  top_k 64, top_p 0.95, **min_p 0** (llama.cpp defaults to 0.05), presence_penalty 0.035 (raised from 0.025 on 2026-11-16 after testing; the Ollama tags carry it), **no repetition penalty**.
  The previous build needed `repeat_penalty 1.06`; this one does not — the SFT, not the sampler, keeps it from
  looping.
- **`--reasoning-budget`** bounds the thinking block (keep `-c` several times larger); the stop phrase tells the model
  to act on what it has. Per-request overrides: `thinking_budget_tokens`, `reasoning_budget_message`
  (llama.cpp ≥ b10091).
- End of turn is token **106** (`<turn|>`), in the GGUF metadata.
- Vision: add `--mmproj mmproj-gemma-4-20B-v9-agentic-F16.gguf`.
- Speculative decoding: the Gemma 4 assistant drafter
  ([`gemma-4-26B-A4B-it-assistant-GGUF`](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-it-assistant-GGUF) Q8_0)
  works with this model on engines with Gemma-4 draft-assistant support (xOllama `mtp-` tags: +60 % tokens/s on a
  3090 at Q4_K_M).

**Ollama.** `mannix/gemma-4-20B-v9-agentic` on ollama.com still serves the **previous build**'s tiers; the sft12
K-FAC tags are published there separately.

## Benchmarks

<!-- OMK:BENCH -->

## Previous build

<!-- OMK:STALE -->

## Lineage

google/gemma-4-26B-A4B-it → [gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic)
(on-policy loop-free SFT, 128 experts) → expert drop to 98/layer with the v9-agentic `ml_k8` competence map, no
shared-expert upweight → **sft12** (short-trace agentic SFT, attention frozen) → F16 → AC imatrix → K-FAC rounding
per tier, QAD at IQ2_XXS. Built with [omk-model-optimizer](https://github.com/mann1x/omk-model-optimizer) and
[omnimergekit](https://github.com/mann1x/omnimergekit).

<sub>Card table rendered <!-- OMK:RENDERED --> by the publisher from the ladder manifest.</sub>
