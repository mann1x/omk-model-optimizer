---
license: apache-2.0
base_model: ManniX-ITA/gemma-4-26B-A4B-agentic
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
quantized_by: ManniX-ITA
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
tags:
- gguf
- llama.cpp
- imatrix
- kfac
- gemma4
- moe
- agentic
- tool-calling
- reasoning
- omk-model-optimizer
---

# Gemma-4-26B-A4B agentic — GGUF (K-FAC quantizations)

GGUF quants of [ManniX-ITA/gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic):
**[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) trained to stop looping on long agentic
tasks** — a BF16 recovery SFT on the model's own regenerated, loop-free answers. Same architecture, tokenizer and chat
template as the base model.

> **Re-published and expanded (2026-11-16).** The three original quants (Q4_K_M, Q6_K, Q8_0, plain `llama-quantize`
> with `imatrix.dat`, llama.cpp b9700) are **replaced by K-FAC rounded files of the same names**, and the ladder is
> extended to every tier in the table. Same model, same sizes, less quantization error.

**Every tier is K-FAC rounded**: same file format, layout and size as the stock `llama-quantize` tier, but the MLP
weights are re-rounded against the model's own curvature, which removes 17–24 % of the quantization error at identical
size on this model (measured below, tier by tier). The 2-bit tier (IQ2_XXS) is a trained student (quantization-aware
distillation) that starts from the K-FAC rounding.

Ollama: [mannix/gemma4-26b-agentic](https://ollama.com/mannix/gemma4-26b-agentic) (plain and `vision-` tags, plus
`mtp-` tags for [xOllama](https://github.com/mann1x/xollama) with the Gemma-4 assistant drafter and a thinking budget;
the head-to-head sampler is preset). The public tags are re-pushed from the K-FAC files tier by tier.

## How the quants are made

Toolchain: [omk-model-optimizer](https://github.com/mann1x/omk-model-optimizer)
([`docs/kfac-rounding.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/kfac-rounding.md);
[the model folder](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-26b-a4b) holds every
script as run), llama.cpp **b12386** for every quantization step.

1. **Base** — the sft11 final weights as a BF16 GGUF (`eos = 106`, official template). The F16 in this repo is the
   b9700 conversion of the same weights.
2. **imatrix** — [`imatrix.gguf`](./imatrix.gguf) (sha256 `80e56b9b…`): the importance matrix the model's own
   QAD run collected from the BF16 teacher on its training corpus — 258 chunks of 16,384 tokens of the sft11 agentic
   data (the 11 families of the weights card), per expert. Every K-FAC tier,
   the IQ2_XXS reference and the IQ2_XXS student share this one file. [`imatrix.dat`](./imatrix.dat) (sha256
   `609a0a04…`, AtomicChat corpus, b9700) is the imatrix of the first publish, kept with its `calibration/` folder for
   the record.
3. **stock tier** — `llama-quantize --imatrix imatrix.gguf` at the tier, with the **Gemma-4 A4B attention rule**: the
   attention projections (`attn_q/k/v/output`) are **Q8_0 from Q4_K_M upward and Q6_K below**. This is the reference
   file: what llama.cpp alone produces; scored in the table as *llama-quantize + imatrix*.
4. **K-FAC rounding** — `omk_qad.kfac_round --stems all --qtype auto --ref <stock tier>`: every MLP tensor (routed
   experts' gate_up and down, dense gate/up/down) is re-rounded by [GPTQ](https://arxiv.org/abs/2210.17323) at the encoder's own 256-column block, with
   the real llama.cpp encoders (byte-identical codes) and the layer's [Kronecker-factored curvature](https://arxiv.org/abs/1503.05671) `H ≈ G ⊗ A`
   (`A` = input second moment, `G` = sampled-label Fisher of the output, collected on 512 windows × 2,048 tokens of the
   same corpus with the attention-sink positions excluded). The output is a byte copy of the stock file with the MLP
   payloads replaced: same tensor types, same byte count, same metadata (imatrix keys included).
5. **twin score** — both files are scored on the QAD cache's 44 held-out windows (614,086 assistant positions) against
   the BF16 teacher's top-64: mean top-256 KL(teacher ‖ quant) and top-1 agreement. The K-FAC file is published only
   when it beats its stock twin; the stock twin is not published.

## Files

<!-- OMK:TIERS -->

**Reading the table.** `KL` = mean KL divergence of the quant's next-token distribution from the BF16 teacher's over
its top tokens, on the held-out assistant positions (lower is better); `top-1` = share of positions where both pick the
same token. `ΔKL` = the K-FAC file's KL relative to its stock twin at identical size. `bpw` = 8 × bytes ÷ the GGUF's
tensor element count. The full sha256 of each file is in its `.gguf.sha256`.

| file | what | size | sha256 |
|---|---|---|---|
| `gemma-4-26B-A4B-agentic-F16.gguf` | unquantised, llama.cpp b9700 conversion | 50.51 GB (47.04 GiB) | `4d640b9d3755e4db…` |
| `mmproj-gemma-4-26B-A4B-agentic-BF16.gguf` | vision projector, BF16 | 1.19 GB (1.11 GiB) | `2aaf3422ef929701…` |
| `imatrix.gguf` | importance matrix of the K-FAC ladder and the QAD (cache corpus, per expert) | 57.0 MB | `80e56b9b597f0af9…` |
| `imatrix.dat` | importance matrix of the first publish (AtomicChat corpus, b9700) | 56.9 MB | `609a0a041ddaa822…` |
| `calibration/` | the first publish's calibration text, manifests and recipe | 17.4 MB | `calibration/SHA256SUMS` |

### IQ2_XXS — the trained 2-bit tier

At 2 bits a rounding is not enough; the IQ2_XXS file is a **quantization-aware distillation** student
([`docs/qad.md`](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/qad.md)): routed experts'
gate_up at IQ2_XXS, routed down at IQ4_NL, dense MLP Q4_K / IQ4_NL, attention Q6_K, embedding Q5_K (layout "E6",
~9.5 GiB). It starts from the **K-FAC rounding of exactly that layout** (`--init-latents`; a seed gate checks the
untrained seeded export against the rounding to 0.0005 KL), then trains the fp16 latents of every quantized tensor
against the BF16 teacher's cached top-64 logits for 600 steps (sequences of 4,096 tokens, lr 2e-6 → 1e-7 cosine, KL
loss, the quantizer in the loop at every step, MTP head distilled with it) on the model's own agentic corpus, and
exports the latents with the same encoders — the file runs on stock llama.cpp.

| arm (same layout, same size) | KL | top-1 |
|---|---:|---:|
| stock `llama-quantize` + imatrix (RTN) | 0.0675 | 0.9392 |
| K-FAC rounding, gate_up only | 0.0576 | 0.9469 |
| K-FAC rounding, all MLP tensors (the seed) | 0.0559 | 0.9480 |
| seeded QAD, step 300 | 0.0493 | — |
| seeded QAD, step 400 | 0.0487 | 0.9492 |
| seeded QAD, step 500 | 0.0483 | 0.9494 |

The published IQ2_XXS is the step-600 export; its final row appears in the table above when it lands.

## Results — head-to-head with the base model

| | base gemma-4-26B-A4B-it | **this model** |
|---|---|---|
| runs finished within the 30-minute limit (20 each) | 3 / 20 (15 %) | **12 / 20 (60 %)** |
| 95 % interval (Wilson) | 5 %–36 % | 39 %–78 % |
| passed within 10 / 20 / 30 min | 0 % / 15 % / 15 % | **25 % / 50 % / 60 %** |
| reasoning-loop cuts per run (median) | 8.0 | **1.0** |
| reasoning-loop cuts per model-hour | 18.1 | **2.9** |

Pass/fail difference: Fisher exact two-sided p = 0.0079.

**What the pass rate means.** It is not an absolute success rate: it is the share of runs that finished the repair
**within a fixed 30-minute budget**. The test is built to measure what the fine-tune changes in a definite amount of
time — how much of the budget goes to loops, how quickly the model gets from a broken file to a working game. A timeout
means "not done in 30 minutes", not "cannot do it". With more time both models would pass more often and the gap would
likely narrow, but not by much: in extended manual testing of the base model and of its pruned 98-expert variants
(the v7-coder and v9-agentic recipes), a run that does not converge quickly becomes steadily more likely to loop, so
the base model either needs one to two hours or more to pass, or ends in massive loops. Anti-repetition sampling hides
the loops but costs the model much of its effectiveness and its ability to converge. The fine-tune targets exactly
this — less looping, faster convergence — and the data below shows it achieved. With only `presence_penalty 0.025`
(no other anti-repetition setting) it is virtually loop-free and completes the task in an acceptable time even on
runs where it does not converge quickly.

**Setup.** One task, run 20 times per model: repair a broken HTML game (`manic_miner`, a syntax error plus logic
bugs). The runs are orchestrated and scored by [automaticum](https://github.com/mann1x/automaticum), a test harness
for coding agents whose oracle runs the game itself. The agent is the
[Cerebriline SDK](https://github.com/mann1x/cline/tree/main/sdk) 4.100.277 (VS Code host profile, LSP diagnostics after every edit
and the `check_file` / `ask_lsp` tools on, browser off) until the game's own oracle reports that it loads, reaches play
and passes 23 logic checks. 30 minutes wall clock per run, one run at a time per model. Both models as BF16 GGUF on the
same llama.cpp build, each on its own RTX PRO 6000 Blackwell (96 GB), f16 KV cache, Gemma-4 assistant drafter for
speculative decoding, thinking on (effort high), temperature 0.8 / top_p 0.95 / top_k 64 / min_p 0 /
presence_penalty 0.025. The two configurations differ only in the model file: harness, task, sampler, server build,
chat template and guards are identical (checked from each run's recorded provenance). The harness guards were on for
both: a reasoning block that repeats itself is cut and the request retried, a repeated identical tool call draws
warnings, then a stop. Arms, server launchers, analysis scripts, the full report and per-run data:
[`omk/models/gemma-4-26b-a4b/sft11/eval/`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-26b-a4b/sft11/eval).

Share of all runs passed by minute *t*: base 0 % / 15 % / 15 %, this model
25 % / 50 % / 60 % at 10 / 20 / 30 min.

Full 1:1 analysis (loops, guards, length, time, tool use, failure modes, known weaknesses):
[weights card](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic#head-to-head-with-the-base-model). Runs on
the BF16 GGUF; the quants are gated by the KL twins above, not by a separate agentic run.

Earlier gate (SDK 4.100.266, temperature 1.0, 10 runs each): base 1 / 10, this model 5 / 10 finished within 30
minutes; loops per model-hour 9.2 → 2.0.

## Use

Standard GGUF types; thinking is controlled by the embedded template (`enable_thinking`). Recommended sampler for
agentic work (the setting used for the head-to-head): temperature 0.8, top_p 0.95, top_k 64, **min_p 0** (llama.cpp
defaults to 0.05), presence_penalty 0.025, no repetition penalty.

```bash
llama-server -m gemma-4-26B-A4B-agentic-Q6_K.gguf --jinja -ngl 99 -fa on -c 65536 \
    --chat-template-kwargs '{"enable_thinking": true}' \
    --temp 0.8 --top-p 0.95 --top-k 64 --min-p 0 --presence-penalty 0.025
```

The tested deployment also used the Gemma-4 assistant drafter
([google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)) for speculative
decoding, which needs an engine with Gemma-4 draft-assistant support.

Ollama: `ollama run mannix/gemma4-26b-agentic` (Q4_K_M; `:Q6_K`, `:Q8_0`, `:vision-<tier>`). The `mtp-<tier>` tags add the drafter and `think_budget high` for [xOllama](https://github.com/mann1x/xollama): [mannix/gemma4-26b-agentic](https://ollama.com/mannix/gemma4-26b-agentic).

Vision: add `--mmproj mmproj-gemma-4-26B-A4B-agentic-BF16.gguf`. The SFT did not touch the vision tower: the projector
is converted from this model's checkpoint (llama.cpp b9700 `convert_hf_to_gguf.py --mmproj --outtype bf16`) and all 356
of its tensors are byte-identical to Google's Gemma-4-26B-A4B-it projector. Vision was not part of the evaluation.

## How it was trained

Built with [omk-model-optimizer](https://github.com/mann1x/omk-model-optimizer) in its **BF16 recovery** mode
([guide](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/bf16-recovery.md)).

- **Data, mostly on-policy**: prompts, tools and recorded tool outputs from 9 NVIDIA Nemotron SFT sets and
  interstellarninja/hermes_reasoning_tool_use; every trained assistant turn regenerated by the untouched
  Gemma-4-26B-A4B-it itself in its own chat template (thinking on) and kept only when a validator found no loop, no
  foreign token and a tool call equivalent to the recorded one. One source regenerated by Gemma-4-31B-it (Kimi-K2.5
  agent tasks). 16,296 conversations → 4,217 windows of 16,384 tokens, one epoch.
  Published: [ManniX-ITA/gemma-4-26B-A4B-agentic-sft](https://huggingface.co/datasets/ManniX-ITA/gemma-4-26B-A4B-agentic-sft),
  [ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b](https://huggingface.co/datasets/ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b).
- **Training**: attention, dense MLP, router and routed experts (24.5 B parameters) in bf16, cross-entropy on the
  regenerated assistant tokens only, 2,108 steps × 2 × 16,384 tokens, peak LR 4.74e-6, one RTX PRO 6000 Blackwell,
  25.5 h.

Recipe, every script as run, corpus provenance and reproduce commands:
[`omk/models/gemma-4-26b-a4b/`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/models/gemma-4-26b-a4b).
The training modes (QAD to GGUF, ternary recovery, BF16 recovery) and the K-FAC rounding are documented in
[`omk/docs/`](https://github.com/mann1x/omk-model-optimizer/tree/omk-main/omk/docs).
More detail on the weights card: [ManniX-ITA/gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic).

The 98-expert prune of this model, retrained for agentic coding on a 16 GB GPU:
[ManniX-ITA/gemma-4-20B-v9-agentic](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic) (same K-FAC ladder).

## Previous files

<!-- OMK:STALE -->

## License

Apache-2.0, as the base model.

<sub>Card table rendered <!-- OMK:RENDERED --> by the publisher from the ladder manifest.</sub>
