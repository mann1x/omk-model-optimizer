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

Every tier is scored twice on the same 252,634 assistant-token positions of the 44 eval rows (11 families), against the
bf16 teacher's top-256 logits: the `llama-quantize` twin (same combined imatrix, same overrides) and the K-FAC file.
Both files of a tier are byte-identical in size. "Rounding error vs RTN" is the K-FAC objective after rounding relative
to round-to-nearest. The ladder is in progress (22 tiers); rows are added as each tier publishes.

<!-- tier table: updated by scripts/omnimerge_v6_kfac/v6_tier_row.py -->
| tier | size (GiB) | llama-quantize + imatrix | **K-FAC** | ΔKL | rounding error vs RTN | tier wall time |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 | 27.05 | 0.00044 / 0.9915 | **0.00043 / 0.9917** | −2.2 % | 0.475x | 56 min |
| Q6_K_L | 21.46 | 0.00074 / 0.9901 | **0.00070 / 0.9902** | −6.3 % | 0.563x | 59 min |
| Q6_K | 20.89 | 0.00081 / 0.9899 | **0.00077 / 0.9899** | −5.4 % | 0.563x | 58 min |
| Q5_K_L | 19.02 | 0.00137 / 0.9873 | **0.00123 / 0.9877** | −9.7 % | 0.601x | 56 min |
| Q5_K_M | 18.29 | 0.00143 / 0.9870 | **0.00130 / 0.9876** | −9.0 % | 0.601x | 57 min |
| Q5_K_S | 17.67 | 0.00157 / 0.9863 | **0.00142 / 0.9873** | −9.5 % | 0.599x | 55 min |
| Q4_K_L | 16.72 | 0.00405 / 0.9796 | **0.00354 / 0.9811** | −12.5 % | 0.610x | 53 min |
| Q4_K_M | 15.84 | 0.00412 / 0.9790 | **0.00362 / 0.9808** | −12.0 % | 0.610x | 58 min |
| IQ4_NL | 15.12 | 0.00496 / 0.9769 | **0.00423 / 0.9791** | −14.8 % | 0.584x | 53 min |
| Q4_K_S | 14.75 | 0.00481 / 0.9776 | **0.00425 / 0.9791** | −11.7 % | 0.608x | 54 min |
| IQ4_XS | 14.48 | 0.00504 / 0.9767 | **0.00431 / 0.9791** | −14.5 % | 0.581x | 53 min |
| Q3_K_XL | 13.80 | 0.01599 / 0.9606 | **0.01192 / 0.9659** | −25.5 % | 0.534x | 54 min |

The gain grows as the bit width falls: at Q8_0 the quantization error is already near the scorer's floor, at Q4_K_M
K-FAC removes an eighth of it.

**Per family, Q4_K_M.** The agentic and code families gain 16–26 % KL; the two largest families (science,
multilingual) gain 5–6 % and hold the pooled figure at −12 %. No family gets worse on either metric.

| family | positions | KL llama-quantize | **KL K-FAC** | ΔKL | top-1 llama-quantize | **top-1 K-FAC** | Δtop-1 (pt) |
|---|---:|---:|---:|---:|---:|---:|---:|
| swe | 2,627 | 0.005310 | **0.003933** | −25.9 % | 97.72 | **97.79** | +0.08 |
| hermes | 12,658 | 0.003031 | **0.002252** | −25.7 % | 98.32 | **98.50** | +0.18 |
| agentic | 12,164 | 0.004312 | **0.003392** | −21.3 % | 97.71 | **98.12** | +0.41 |
| kimi31b | 6,369 | 0.005016 | **0.004043** | −19.4 % | 96.80 | **97.33** | +0.53 |
| cp | 42,023 | 0.003072 | **0.002500** | −18.6 % | 98.20 | **98.45** | +0.25 |
| opencode | 2,850 | 0.005124 | **0.004292** | −16.2 % | 97.51 | **97.82** | +0.32 |
| ifchat_v1 | 11,374 | 0.007101 | **0.006086** | −14.3 % | 96.60 | **96.78** | +0.18 |
| math | 16,691 | 0.002843 | **0.002473** | −13.0 % | 98.51 | **98.66** | +0.16 |
| ifchat_v2 | 47,174 | 0.004916 | **0.004332** | −11.9 % | 97.66 | **97.88** | +0.22 |
| science | 49,719 | 0.003814 | **0.003591** | −5.9 % | 98.04 | **98.08** | +0.04 |
| multilingual | 48,985 | 0.004280 | **0.004048** | −5.4 % | 97.94 | **98.05** | +0.11 |
| **all** | **252,634** | **0.004116** | **0.003621** | **−12.0 %** | **97.90** | **98.08** | **+0.18** |

Source: `runs/v6_kfac/ladder/<tier>/{ptq,kfac}.eval.json` (the twin scorer's pooled and `families` blocks).

## Known gaps

- Only the MLP is re-rounded; the gated-deltanet and attention projections keep the `llama-quantize` rounding.
- All 22 published tiers are K-FAC rounded, the `_L` (embed/output Q8_0), Q3_K_XL/L/S and Q2_K tiers included;
  the non-MLP tensors of every tier keep the stock rounding with the combined imatrix.

## Cards

- `cards/v6_gguf_card.md` — the GGUF repo card with the K-FAC section; **held**, uploaded to HF when the ladder completes.
- `cards/v6_gguf_card.hf_live.md` — snapshot of the live HF card (2026-11-28) it was edited from.
