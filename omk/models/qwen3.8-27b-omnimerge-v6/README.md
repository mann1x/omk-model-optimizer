# Qwen3.8-27B OmniMerge v6 — the first dense / hybrid model through the K-FAC ladder

**Published:** [ManniX-ITA/Qwen3.8-27B-Omnimerge-v6](https://huggingface.co/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6)
(BF16 safetensors) · [ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF](https://huggingface.co/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF)
(K-FAC ladder, MTP head kept, vision `mmproj`) · Ollama `mannix/omnimerge-v6` (text and `vision-` tags).

**What it is.** An [OmniMergeKit](https://github.com/mann1x/omnimergekit) merge on `Qwen/Qwen3.8-27B` (task base
`Qwen/Qwen3.6-27B`) of three Qwen3.6-27B reasoning fine-tunes, MLP and MTP head passed through from Qwen3.8 (the merge
recipe is on the model card). Architecture `qwen3_5` / llama.cpp `qwen35`: 64 decoder layers, three gated-deltanet
(linear attention) layers for every full-attention layer, hidden 5120, dense MLP 17408, vocabulary 248,320, one MTP layer
(`blk.64` nextn), untied embeddings, no logit softcap.

This page is about its **requantization**: the published `llama-quantize` ladder (AtomicChat imatrix, 2026-09-09) was
rebuilt with [K-FAC rounding](../../docs/kfac-rounding.md) — the first model of the program without experts.

| | |
|---|---|
| quantization | [K-FAC rounding](../../docs/kfac-rounding.md) of the dense MLP (`gate`/`up`/`down`, 64 layers) of every tier; combined AtomicChat + QAD-cache imatrix |
| hosts | bs2, one RTX PRO 6000 Blackwell 96 GB per job (GPU0; GPU1 for a third ladder lane) |
| scripts | [`quant/`](quant/) (copies of the run scripts) |

## 1. What changed in the code for a dense model

- `omk_qad.kfac` collected factors only on Gemma-4 MoE layers. It now has a dense-only path: a layer without a MoE block
  gets the dense MLP accumulators alone (`--dense`), the head no longer assumes a logit softcap, and the self-check knows
  the dense case. Tested on a tiny hybrid Qwen3.5 (`omk_qad/tests/test_kfac_dense_cpu.py`): the hooked gate/up/down
  gradients reproduce autograd's weight gradients on a linear-attention layer and on the full-attention layer.
- `--probe-grow N`: the attention-sink cut of `G` (`--drop-grow`) is an absolute gradient norm calibrated on Gemma (10).
  On a new model a probe pass measures the dense gate‖up gradient row norms and the cut is set to their pooled 99th
  percentile — the same "about 1 % of rows" regime the Gemma cut produced.
- `omk_qad.kfac_round --stems dense`.
- Every long stage can be paused and resumed: the QAD cache between shards, the factors between passes, the rounder
  between layers (`--control`; the output is written to `OUT.partial` with a per-layer progress file and appears only
  when complete; a drained run exits 75 and the same command resumes).

## 2. Calibration data

The 11 conversation families of the Gemma-4 agentic program, template-free
([ManniX-ITA/gemma-4-26B-A4B-agentic-sft](https://huggingface.co/datasets/ManniX-ITA/gemma-4-26B-A4B-agentic-sft)),
**re-rendered with the v6 tokenizer and chat template** (`omk_qad.rollouts --history-reasoning strip --seq 16384 --pack`:
assistant-only role masks, earlier turns' reasoning stripped as the template does, 16k packed windows), weighted as the
26B/20B cache: agentic 14.87, opencode 22.90, ifchat_v1 5.73, hermes 3.29, ifchat_v2 2.84, kimi31b 2.64, swe 1.03,
multilingual 0.62, math 0.29, science 0.26, cp 0.19. The older Qwen render of this repository (`mix_v1/v2`, Qwen3.5-4B
tests) has no role masks, 4k windows and lacks hermes, ifchat, kimi31b, math, science and multilingual, so it was not
reused. The assistant turns were written by Gemma-4: they are context only — every label (the cache's top-64, the
factors' sampled tokens) comes from OmniMerge v6 itself.

AtomicChat imatrix: `AtomicChat/calib-corpora` `builds/qwen3.8-27b/calib_train.txt`, all 9,703 chunks (4.97 M tokens,
`--parse-special`), the `imatrix.dat` already on the GGUF repo (sha256 `50f44d8f…`).

## 3. Recipe

1. QAD cache: 768 × 16,384-token windows of the mix (teacher top-64), 4 eval rows per family (44 rows), cache imatrix on
   256 windows.
2. Imatrix: `llama-imatrix --in-file imatrix.dat --in-file cache/imatrix.gguf` (= `omk_qad.train --imatrix-combine`,
   [§4.0.1](../../docs/kfac-rounding.md)).
3. Gate: the BF16 GGUF scored against the HF teacher on the eval rows (KL must be ~0).
4. Factors: 512 windows × 2048 tokens, seed 1, `--dense --down-g diag --drop-grow <probe p99>`.
5. Each tier: `llama-quantize --imatrix <combined>` twin with the published v6 overrides (MTP head `blk.64` at Q4_K for
   IQ2_*, IQ3_XXS, IQ3_XS), then `kfac_round --stems dense --qtype auto --ref twin`, both scored (KL / top-1 vs the bf16
   teacher, top-256). Attention, linear attention, embeddings and the MTP head keep the twin's bytes.

### Preparation time (once per model)

Clean run on bs2 (one RTX PRO 6000 96 GB, 16 CPU threads); failed attempts and idle gaps excluded. Each tier then takes
56–58 min end to end (PTQ + K-FAC rounding + twin score + HF/ollama publish); see
[kfac-rounding §6.3](../../docs/kfac-rounding.md#63-a-dense-model-the-omnimerge-v6-ladder-qwen38-27b-tier-by-tier-and-per-family).

| stage | device | wall time |
|---|---|---:|
| render 11 corpus families (train + eval) | CPU | 7.1 min |
| layout + HF → BF16 GGUF convert (54.7 GB) | CPU | 5.7 min |
| AC imatrix (`llama-imatrix`, AtomicChat corpus, 9,703 chunks x 512 = 5.0 M tokens, `--parse-special`) | GPU | 4 h 23 min — **reused** here from the 2026-09 stock ladder; a model without one pays it once |
| AC imatrix re-save + fused-name remap + check | CPU | 2.5 min |
| **QAD cache**: 768 x 16k windows, teacher top-64 + 4 eval rows per family + cache imatrix | GPU | **82 min** (12.7 s/row for the first 256 rows while the imatrix is collected, 3.25 s/row after) |
| combined imatrix (AC + cache) + check | CPU | < 1 min |
| base-GGUF gate (BF16 GGUF vs the HF teacher, 44 eval rows) | GPU | 7.2 min |
| grow probe (64 windows x 2048) | GPU | 1.6 min |
| **K-FAC dense factors**: 64 layers x 512 windows x 2048, 3 layers/pass | GPU | **3.0 h** (0.96 s/window, peak 92.0 GiB) |
| **total before the first tier** | | **≈ 4.5 h GPU + 16 min CPU** with the AC imatrix in hand; **≈ 8.9 h GPU** from scratch |

## 4. Results

<!-- results: filled from ladder/twins.log when the ladder completes -->
_Pending (run started 2026-11-28 00:00 CET)._

## Known gaps

- Only the MLP is re-rounded; the gated-deltanet and attention projections keep the `llama-quantize` rounding.
- The tiers outside the K-FAC set (Q6_K_L, Q5_K_L, Q4_K_L, Q3_K_XL, Q3_K_L, Q3_K_S, Q2_K) stay the stock AC-imatrix files.

## Cards

- `cards/v6_gguf_card.md` — the GGUF repo card with the K-FAC section; **held**, uploaded to HF when the ladder completes.
- `cards/v6_gguf_card.hf_live.md` — snapshot of the live HF card (2026-11-28) it was edited from.
