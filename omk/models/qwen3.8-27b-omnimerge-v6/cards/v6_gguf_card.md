---
license: apache-2.0
base_model:
- ManniX-ITA/Qwen3.8-27B-Omnimerge-v6
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
tags:
- gguf
- imatrix
- mtp
- quantized
- speculative-decoding
- vision
model_name: Qwen3.8-27B-Omnimerge-v6-MTP-GGUF
author: ManniX-ITA
architecture: Qwen3.5
parameters: 27.78B
format: GGUF
quantizations:
- F16
- IQ2_M
- IQ2_S
- IQ3_M
- IQ3_XS
- IQ3_XXS
- IQ4_NL
- IQ4_XS
- Q2_K
- Q3_K_L
- Q3_K_M
- Q3_K_S
- Q3_K_XL
- Q4_K_L
- Q4_K_M
- Q5_K_L
- Q5_K_M
- Q6_K
- Q6_K_L
- Q8_0
features:
- MTP
- imatrix
- merge
- reasoning
- vision
---

# Qwen3.8-27B-Omnimerge-v6-MTP-GGUF

GGUF quantizations of [`ManniX-ITA/Qwen3.8-27B-Omnimerge-v6`](https://huggingface.co/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6) — a task-arithmetic merge of three Qwen3.6 fine-tunes onto the **Qwen3.8-27B** base — with the **MTP head retained** (`blk.64.*`, 866 tensors) and the F16 vision projector published alongside.

19 tiers. `imatrix.dat` is archived in this repo, so the imatrix tiers are reproducible and auditable.

## Available Quantizations

| Tier | Size ≈ | imatrix | Notes |
|---|---|---|---|
| Q8_0 | 29.1 GB | no | near-lossless |
| Q6_K_L | 23.1 GB | no | Q6_K with F16 output/embed |
| **Q6_K** | 22.4 GB | no | **the eval reference tier** — every benchmark below |
| Q5_K_L | 20.3 GB | no | |
| Q5_K_M | 19.5 GB | no | |
| Q4_K_L | 17.8 GB | no | |
| **Q4_K_M** | 16.8 GB | no | ollama `:latest`; 24 GB cards |
| IQ4_NL | 16.0 GB | yes | |
| IQ4_XS | 15.3 GB | yes | smallest 4-bit |
| Q3_K_XL | 14.6 GB | yes | |
| Q3_K_L | 14.6 GB | yes | |
| Q3_K_M | 13.5 GB | yes | |
| IQ3_M | 12.8 GB | yes | |
| Q3_K_S | 12.3 GB | yes | |
| IQ3_XS | 12.2 GB | yes | |
| IQ3_XXS | 11.4 GB | yes | |
| Q2_K | 11.0 GB | yes | |
| IQ2_M | 10.2 GB | yes | |
| IQ2_S | 9.6 GB | yes | smallest shipped; 12 GB cards |

### Tensor names: re-download a file fetched on 2026-11-28 before the fix

The K-FAC files are built with llama.cpp b12386. For this architecture (`qwen35`) its converter stores two pairs of
linear-attention projections of every layer as one tensor each: `attn_qkv_gate` (= `attn_qkv` + `attn_gate`) and
`ssm_beta_alpha` (= `ssm_beta` + `ssm_alpha`). Older loaders do not know those names and refuse the file with

```
llama_model_load: error loading model: missing tensor 'blk.0.ssm_beta.weight'
```

That covers older llama.cpp builds and older ollama / ollama-based runtimes, including the opencoti engine up to c14.

The first five K-FAC tiers were uploaded in that merged form on 2026-11-28: Q4_K_M, Q6_K, Q8_0, Q5_K_M and Q4_K_L,
together with their ollama tags. All of them were replaced the same day by files with the classic names.
- The two tensors of each pair are split back apart. That is a byte copy of whole rows: same quantization, same
  weights. Only the file's sha256 changes; see each file's `.sha256` sidecar.
- Every file in this repo and every `mannix/omnimerge-v6` tag now uses the split names, which old and new loaders
  both read.
- The split files score the same as the merged ones within noise: pooled KL 0.0036199 vs 0.0036210 on Q4_K_M.

If a download or `ollama pull` from 2026-11-28 fails with the error above, fetch it again.

To check a file, list its tensor names (`gguf-dump FILE | grep -c ssm_beta_alpha`). The count must be 0.

### Why not every tier uses imatrix

This is a measured policy, not an omission. The imatrix crossover is a property of the quant **grid coarseness**: at the Q4 band and above the calibration bias outweighs importance-weighting and imatrix is neutral-to-harmful; below it (Q3/Q2/IQ) imatrix is the difference between a good quant and a broken one. Measured once per model family on a Q2→Q6 HE+/MPE ladder. So Q4_K_*/Q5_K_*/Q6_K* are built **without** imatrix and everything at Q3 and below — plus the IQ tiers — **with** it. Q8_0 needs none.

## How to Use — MTP speculative decoding

Requires llama.cpp containing [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) ("llama + spec: MTP Support"). Older builds load the weights and silently ignore the `mtp.*` head — standard decode, no error, no speedup.

```bash
llama-server -m Qwen3.8-27B-Omnimerge-v6-Q6_K.gguf \
    -c 16384 -ngl 99 \
    --parallel 1 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --reasoning-format deepseek --reasoning-budget 8192 \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --port 8099
```

- `--spec-type draft-mtp` — enables MTP self-speculative decoding using the included head.
- `--spec-draft-n-max 3` — tokens proposed per step; 3 is the sweet spot on this family.
- Omit `--spec-type` to run as a plain release; quality is unaffected either way.

> The MTP head here is **Qwen3.8's own, preserved verbatim** — it is deliberately excluded from the merge, because applying a 3.6-trained head's delta to 3.8's head degrades draft acceptance silently.

### Measured MTP speedup

Q4_K_M on an RTX PRO 6000 Blackwell, llama.cpp b9700, `-c 8192 -ngl 99 --parallel 1`, default f16 KV, greedy, 5 varied prompts × 512 generated tokens:

| arm | median tok/s | speedup | draft acceptance | mean accepted length |
|---|---|---|---|---|
| MTP off | 69.4 | 1.00× | — | — |
| `--spec-draft-n-max 2` | 130.3 | 1.88× | 0.81 | 2.62 |
| **`--spec-draft-n-max 3`** | **140.4** | **2.02×** | 0.73 | 3.19 |
| `--spec-draft-n-max 4` | 134.0 | 1.93× | 0.61 | 3.43 |

**n=3 is the sweet spot at 2.0×.** Raising `n-max` lengthens the mean accepted run (2.62 → 3.19 → 3.43) but lowers the acceptance *rate* (0.81 → 0.73 → 0.61); by n=4 the extra verification costs more than the longer runs return. The off arm is very stable (68.7–69.5 tok/s across all five prompts); the MTP arms spread far wider (113–170) because acceptance depends on what is being generated — take the median, not a single prompt.

> **MTP output is not bit-identical to MTP-off on this model.** Verified with a same-arm control: the off arm run twice is byte-identical, so the rig is deterministic — but MTP-on diverged from MTP-off on 4 of 5 prompts at 512 tokens. Speculative decoding evaluates the target on an N+1-token batch rather than batch-1, and the different floating-point reduction order flips greedy `argmax` at near-ties. MTP stays *algorithmically* lossless (it only accepts tokens the target would emit) and no quality difference is implied — but quality equivalence was **not** measured here, so treat this as "2× throughput, outputs may differ at near-ties", not "identical output".

### Reasoning budget

`--reasoning-budget` matters more here than on v4. v6 moves work out of the answer channel into the reasoning channel (GPQA reasoning median 19,488 chars vs v4's 12,010), and **6 of 198 GPQA questions scored zero purely by running past an ~8192-token thinking budget**, where v4 had none. Budget generously.

## Vision / multimodal

The projector is published in this repo: `mmproj-Qwen3.8-27B-Omnimerge-v6-F16.gguf` (~0.93 GB, F16). All 333 `model.visual.*` tensors are bit-identical to the Qwen3.8 base, so it is the stock tower.

```bash
llama-server -m Qwen3.8-27B-Omnimerge-v6-Q6_K.gguf \
    --mmproj mmproj-Qwen3.8-27B-Omnimerge-v6-F16.gguf \
    -c 16384 -ngl 99 --port 8099
```

Companion `vision-<tier>` ollama tags bundle the projector with each text tier; rollout is in progress alongside the quant ladder.

## ollama

[`mannix/omnimerge-v6`](https://ollama.com/mannix/omnimerge-v6) — `:latest` = Q4_K_M.

```bash
ollama run mannix/omnimerge-v6
ollama run mannix/omnimerge-v6:Q6_K      # the eval reference tier
ollama run mannix/omnimerge-v6:IQ2_S     # smallest, fits 12 GB
```

Every tag carries the **same serving identity as the official `qwen3.8:27b`**, copied verbatim from its published manifest rather than guessed:

```
RENDERER  qwen3.8
PARSER    qwen3.5        # note: ollama has no "qwen3.8" parser
PARAMETER draft_num_predict 4 · min_p 0 · presence_penalty 0
          repeat_penalty 1 · temperature 1 · top_k 20 · top_p 0.95
```

Because the tags declare `RENDERER qwen3.8`, they require **ollama >= 0.32.12** (0.33.x recommended). On older builds the model downloads, loads, and reports healthy VRAM in `ollama ps`, then fails at prompt-render time with `unknown renderer "qwen3.8"` — upgrade ollama; nothing is wrong with the model.

`num_ctx` is deliberately not pinned. The architecture is good for 256k natively and only 16 of the 64 layers carry a KV cache (the other 48 are gated-SSM), so context is cheap here — but ollama's default is far below what this model wants:

```
ollama run mannix/omnimerge-v6
>>> /set parameter num_ctx 65536
```

## K-FAC rounding — how these files were built

The MLP tensors of each `llama-quantize` tier are re-rounded against the model's Kronecker-factored curvature with
llama.cpp's own encoders, so each file keeps the stock layout, size and metadata
([method and results](https://github.com/mann1x/omk-model-optimizer/blob/omk-main/omk/docs/kfac-rounding.md#63-a-dense-model-the-omnimerge-v6-ladder-qwen38-27b-tier-by-tier-and-per-family)).

### Preparation time (once per model)

Clean run on one RTX PRO 6000 96 GB with 16 CPU threads; failed attempts and idle gaps excluded. Each tier then takes
56–58 min end to end (PTQ + K-FAC rounding + twin score + HF/ollama publish).

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

## Benchmark Results

Sampled cohort (`recommended`: temperature 0.6 / top-p 0.95 / top-k 20, `do_sample=true`),
Q6_K, `llama.cpp`, identical benches and sampler across every cell.

> Comparable **to each other only** — do not pool with greedy-decode results from any
> other card. The greedy GPQA table further down is a **separate cohort**: its rows must
> never be merged into these.

### 1. Calibration change — previous imatrix vs AtomicChat

| Benchmark | previous imatrix | **AtomicChat imatrix** | Δ |
|---|---|---|---|
| HumanEval (thinking) | 0.982 | 0.976 | -0.60 pp |
| IFEval (100) | 0.960 | 0.940 | -2.00 pp |
| **LiveCodeBench (v6, 77q)** | 0.883 | 0.883 | ±0.00 |
| MultiPL-E (100) | 0.887 | 0.893 | +0.60 pp |
| **Mean (4)** | 0.928 | 0.923 | -0.50 pp |

The published ladder had been calibrated on a 128-chunk in-repo corpus, and six K-tiers
(Q6_K_L, Q6_K, Q5_K_L, Q5_K_M, Q4_K_L, Q4_K_M) carried **no imatrix at all** — an
exclusion policy measured on a different model family that should never have applied
here. (Q8_0 also carries none, but that is correct: it is imatrix-free by rule.)
Every `_K`/`IQ` tier is being rebuilt on
[AtomicChat/calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora)
`builds/qwen3.8-27b` at the **full 9,703 chunks**, ~76× the previous basis. The Q6_K
above is the recalibrated file; the remaining tiers are re-uploading and this note will
be removed when the ladder is complete.

Read these four deltas as a wash, not a win or a loss. The two negatives (IFEval −2.00 pp
= 2 questions of 100; HumanEval −0.60 pp = 1 of 164) and the one positive (MultiPL-E
+0.60 pp) are all inside single-draw sampled noise, and LiveCodeBench is unchanged to four
decimals. The case for recalibrating is not a score gain — it is that seven tiers were
shipping uncalibrated, and that an omitted imatrix is the **asymmetric** error: at worst
it costs ~1 pp where it is neutral, while omitting one where it matters is catastrophic.

### 2. Omnimerge-v6 vs Omnimerge-v4

| Benchmark | **Omnimerge-v6 (3.8)** | Omnimerge-v4 (3.6) | Δ |
|---|---|---|---|
| GPQA-Diamond (198q) | 0.768 | 0.788 | -2.00 pp |
| HumanEval (thinking) | 0.982 | 0.982 | ±0.00 |
| IFEval (100) | 0.960 | 0.950 | +1.00 pp |
| **LiveCodeBench (v6, 77q)** | 0.883 | 0.818 | +6.50 pp |
| MultiPL-E (100) | 0.887 | 0.877 | +1.00 pp |
| **Mean (5)** | 0.896 | 0.883 | +1.30 pp |

This table is kept **whole and unchanged** on the previous-imatrix Q6_K column, because
it is a controlled v6-vs-v4 pair. Swapping some cells to the recalibrated quant while
others stayed would make the row set mixed-basis, which is worse than slightly stale; the
calibration change is measured separately in table 1.

Only LiveCodeBench is outside the measurement band (+6.5 pp = 5 more solved of 77). The
±1 pp cells are **not** differences — a same-basis MultiPL-E-100 repeat on this rig moved
1.0 pp from batch scheduling alone. GPQA's −2 pp is 4 questions of 198: suggestive, not
established. The mean is carried almost entirely by LiveCodeBench.

### 3. GPQA — greedy only, paired

For the **calibration** question, GPQA is reported only from a greedy paired run, and is
deliberately absent from table 1. Its sampled cell there was a single noisy draw reading
−5.05 pp for the AtomicChat quant; a paired greedy rerun of that identical pair read
**+2.02 pp** — the opposite sign, so the sampled number would have misrepresented the
change. (GPQA remains in table 2, where both columns sit on the same sampled basis and the
comparison is v6-vs-v4 rather than imatrix-vs-imatrix.) The greedy pair is the trustworthy measurement: both arms ran the same
morning, concurrently, on the same host and binary, scored per-question on the same 198
items.

| GPQA-Diamond (198q), greedy | previous imatrix | **AtomicChat imatrix** |
|---|---|---|
| flexible-extract | 0.7778 (154/198) | **0.7980 (158/198)** |

```
              ac_correct  ac_wrong          delta(ac-prev)  = +2.02 pp
prev_correct     143         11             McNemar exact p = 0.5572
prev_wrong        15         29             95% CI on delta = [-3.02, +7.06] pp
                                            discordant      = 26/198 = 13.1%
```

p = 0.5572: the two quants are **statistically indistinguishable** on GPQA. This is not a
win. Its value is the interval — it **excludes −5 pp**, so the degradation the sampled
cell suggested is ruled out at 95%. Degenerate outputs (empty or `[invalid]`) were 5 for
the previous imatrix vs 2 for AtomicChat, and response lengths are comparable (p50 1142 vs
1187 chars), so nothing is hiding under the equal means.

## Known Limitations

- **Verbose.** See the reasoning-budget note above; this is the main practical difference from v4.
- **MTP draft-n is hardware-dependent.** ollama *does* run the MTP head (see the ollama section above); these tags ship `draft_num_predict 4`. The optimum is a property of the weights × the GPU, not of MTP — on a sibling A3B, n=3 gave +33 % on Blackwell while n=8 was **−23 %, worse than not speculating at all**. Re-measure before lifting an `n` from another model or card.
- **Vision + MTP together is not validated.** Both flags load under llama.cpp, but draft acceptance with vision tokens in the prompt is unmeasured. Drop `--spec-type` if you hit trouble.
- Benchmarks are Q6_K only; lower tiers are not separately evaluated.
- Research checkpoint: one cohort, one seed, one tier.

---

## Original Model Card


# Qwen3.8-27B-Omnimerge-v6

Task-arithmetic merge of three Qwen3.6 fine-tunes onto the **Qwen3.8-27B** base — the same sources, weights and method as [`ManniX-ITA/Qwen3.6-27B-Omnimerge-v4`](https://huggingface.co/ManniX-ITA/Qwen3.6-27B-Omnimerge-v4), moved to the newer base generation. No fine-tuning and no distillation of its own: weight-space arithmetic over published checkpoints only.

Named v6 because v5 is taken by `Qwen3.6-27B-Omnimerge-v5-mlp-skip`.

> **Headline:** on a five-bench sampled cohort run head-to-head against v4 on the same binary and sampler, the one result outside the measurement band is **LiveCodeBench 0.883 vs 0.818 (+6.5 pp, 68/77 vs 63/77)**. Everything else ties or sits inside noise; GPQA is 2 pp lower. v6 also thinks substantially longer for it — see [Thinking profile](#thinking-profile).

## Sources

| Source | Weight | Role |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | base | base, tokenizer, chat template, MTP head, vision tower |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | task base | deltas are computed **from here** |
| [rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled](https://huggingface.co/rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled) | 0.40 | general capability |
| [ValiantLabs/Qwen3.6-27B-Esper3.1](https://huggingface.co/ValiantLabs/Qwen3.6-27B-Esper3.1) | 0.35 | code + reasoning |
| [kai-os/Qwen3.6-27b-Opus4.6-reasoning](https://huggingface.co/kai-os/Qwen3.6-27b-Opus4.6-reasoning) | 0.25 | reasoning anchor (LoRA applied to Qwen3.6, then merged) |

Method: `omnimerge_v2` (DARE-TIES base + OBIM-lite + DAREx q + EMR election). Density 0.53, DAREx q 0.75, seed 42. `mlp.gate_proj / up_proj / down_proj` and `mtp.` are passed through from the base.

## The task base is load-bearing

Every source was fine-tuned **from Qwen3.6**, so each delta must be taken against 3.6 and then applied to 3.8. Merging without `--task-base` computes `delta = source − 3.8`, which embeds the *inverse* of the 3.6→3.8 generational upgrade and drags the result backwards — and on the 7 embedding rows 3.8 adds, it would compute `(3.6 padding) − (3.8 audio embedding)` and corrupt them.

Proof the transplant landed, from `verify_merge_artifact.py` on a sampled `q_proj`:

```
||out − Qwen3.8|| =  1.09      <- output sits on the 3.8 base
||out − Qwen3.6|| = 67.78
```

## Verified, not assumed

**Vocabulary.** `vocab.json` and `merges.txt` are sha256-identical between Qwen3.6-27B and Qwen3.8-27B. 3.8 adds 7 tokens **purely additively** at ids 248070–248076 (audio/TTS); no id is reassigned or dropped. All three sources match the 3.8 vocab exactly — their `merges` differ only in serialisation (`"Ġ Ġ"` vs `["Ġ","Ġ"]`).

**Vision tower.** All **333** `model.visual.*` tensors (167 weights + 166 non-weights) are bit-identical between the merge and the 3.8 base, so the stock Qwen3.8 projector is the correct `mmproj`.

**MTP head is deliberately not merged.** All three sources carry the 15 `mtp.*` tensors, so the default would apply a 3.6-trained head's delta to 3.8's head. MTP affects draft-acceptance rate only — it costs decode speed, silently. 3.8's head is preserved verbatim and appears in the GGUF as `blk.64.*` (blocks 0..64, 866 tensors). `MERGE_MTP=1` merges it instead.

## Benchmark Results

Sampled cohort (`recommended`: temperature 0.6 / top-p 0.95 / top-k 20, `do_sample=true`),
Q6_K, `llama.cpp`, identical benches and sampler across every cell.

> Comparable **to each other only** — do not pool with greedy-decode results from any
> other card. The greedy GPQA table further down is a **separate cohort**: its rows must
> never be merged into these.

### 1. Calibration change — previous imatrix vs AtomicChat

| Benchmark | previous imatrix | **AtomicChat imatrix** | Δ |
|---|---|---|---|
| HumanEval (thinking) | 0.982 | 0.976 | -0.60 pp |
| IFEval (100) | 0.960 | 0.940 | -2.00 pp |
| **LiveCodeBench (v6, 77q)** | 0.883 | 0.883 | ±0.00 |
| MultiPL-E (100) | 0.887 | 0.893 | +0.60 pp |
| **Mean (4)** | 0.928 | 0.923 | -0.50 pp |

The published ladder had been calibrated on a 128-chunk in-repo corpus, and six K-tiers
(Q6_K_L, Q6_K, Q5_K_L, Q5_K_M, Q4_K_L, Q4_K_M) carried **no imatrix at all** — an
exclusion policy measured on a different model family that should never have applied
here. (Q8_0 also carries none, but that is correct: it is imatrix-free by rule.)
Every `_K`/`IQ` tier is being rebuilt on
[AtomicChat/calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora)
`builds/qwen3.8-27b` at the **full 9,703 chunks**, ~76× the previous basis. The Q6_K
above is the recalibrated file; the remaining tiers are re-uploading and this note will
be removed when the ladder is complete.

Read these four deltas as a wash, not a win or a loss. The two negatives (IFEval −2.00 pp
= 2 questions of 100; HumanEval −0.60 pp = 1 of 164) and the one positive (MultiPL-E
+0.60 pp) are all inside single-draw sampled noise, and LiveCodeBench is unchanged to four
decimals. The case for recalibrating is not a score gain — it is that seven tiers were
shipping uncalibrated, and that an omitted imatrix is the **asymmetric** error: at worst
it costs ~1 pp where it is neutral, while omitting one where it matters is catastrophic.

### 2. Omnimerge-v6 vs Omnimerge-v4

| Benchmark | **Omnimerge-v6 (3.8)** | Omnimerge-v4 (3.6) | Δ |
|---|---|---|---|
| GPQA-Diamond (198q) | 0.768 | 0.788 | -2.00 pp |
| HumanEval (thinking) | 0.982 | 0.982 | ±0.00 |
| IFEval (100) | 0.960 | 0.950 | +1.00 pp |
| **LiveCodeBench (v6, 77q)** | 0.883 | 0.818 | +6.50 pp |
| MultiPL-E (100) | 0.887 | 0.877 | +1.00 pp |
| **Mean (5)** | 0.896 | 0.883 | +1.30 pp |

This table is kept **whole and unchanged** on the previous-imatrix Q6_K column, because
it is a controlled v6-vs-v4 pair. Swapping some cells to the recalibrated quant while
others stayed would make the row set mixed-basis, which is worse than slightly stale; the
calibration change is measured separately in table 1.

Only LiveCodeBench is outside the measurement band (+6.5 pp = 5 more solved of 77). The
±1 pp cells are **not** differences — a same-basis MultiPL-E-100 repeat on this rig moved
1.0 pp from batch scheduling alone. GPQA's −2 pp is 4 questions of 198: suggestive, not
established. The mean is carried almost entirely by LiveCodeBench.

### 3. GPQA — greedy only, paired

For the **calibration** question, GPQA is reported only from a greedy paired run, and is
deliberately absent from table 1. Its sampled cell there was a single noisy draw reading
−5.05 pp for the AtomicChat quant; a paired greedy rerun of that identical pair read
**+2.02 pp** — the opposite sign, so the sampled number would have misrepresented the
change. (GPQA remains in table 2, where both columns sit on the same sampled basis and the
comparison is v6-vs-v4 rather than imatrix-vs-imatrix.) The greedy pair is the trustworthy measurement: both arms ran the same
morning, concurrently, on the same host and binary, scored per-question on the same 198
items.

| GPQA-Diamond (198q), greedy | previous imatrix | **AtomicChat imatrix** |
|---|---|---|
| flexible-extract | 0.7778 (154/198) | **0.7980 (158/198)** |

```
              ac_correct  ac_wrong          delta(ac-prev)  = +2.02 pp
prev_correct     143         11             McNemar exact p = 0.5572
prev_wrong        15         29             95% CI on delta = [-3.02, +7.06] pp
                                            discordant      = 26/198 = 13.1%
```

p = 0.5572: the two quants are **statistically indistinguishable** on GPQA. This is not a
win. Its value is the interval — it **excludes −5 pp**, so the degradation the sampled
cell suggested is ruled out at 95%. Degenerate outputs (empty or `[invalid]`) were 5 for
the previous imatrix vs 2 for AtomicChat, and response lengths are comparable (p50 1142 vs
1187 chars), so nothing is hiding under the equal means.

## Quantizations

**GGUF (llama.cpp / ollama / text-generation-webui)** — 19 tiers with the MTP head retained, plus the F16 vision projector:
[`ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF`](https://huggingface.co/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF)

**ollama** — [`mannix/omnimerge-v6`](https://ollama.com/mannix/omnimerge-v6) (`:latest` = Q4_K_M). Tags carry the same serving identity as the official `qwen3.8:27b` (`RENDERER qwen3.8`, `PARSER qwen3.5`, vendor sampling defaults), so they need **ollama >= 0.32.12**.

```bash
ollama run mannix/omnimerge-v6
```

## Reproducing the merge

```bash
python omnimergekit.py \
    --base      Qwen/Qwen3.8-27B \
    --task-base Qwen/Qwen3.6-27B \
    --source    rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled \
    --source    ValiantLabs/Qwen3.6-27B-Esper3.1 \
    --source    <kai-os LoRA applied to Qwen3.6-27B> \
    --weights 0.40,0.35,0.25 \
    --method omnimerge_v2 \
    --density 0.53 --darex-q 0.75 --seed 42 \
    --skip-patterns 'mlp.gate_proj,mlp.up_proj,mlp.down_proj,mtp.' \
    --output Qwen3.8-27B-Omnimerge-v6
```

The kai-os source is a LoRA; it is applied to Qwen3.6-27B first (`scripts/apply_lora_to_safetensors.py`) and the resulting anchor is merged as a full model. Tokenizer, chat template and preprocessor configs are copied from the 3.8 base after the merge.

Merge engine: [mann1x/omnimergekit](https://github.com/mann1x/omnimergekit).

## Caveats

- **Research checkpoint.** One eval cohort, one seed, one quantization tier (Q6_K). The only delta established beyond the noise band is LiveCodeBench.
- **Verbose.** Budget more thinking headroom than you would for v4; six GPQA questions scored zero on budget exhaustion alone.
- Benchmarks were measured on Q6_K. Lower tiers are not separately evaluated.
- Sampled-cohort numbers throughout — not comparable to greedy tables elsewhere.

## Acknowledgements

Qwen team for the Qwen3.8 base and the vision/MTP components; rico03, ValiantLabs and kai-os for the source fine-tunes.

## Tool-calling benchmark — `tool-eval-bench` hardmode (88 scenarios, 176 pts)

**Benchmarked file: `Qwen3.8-27B-Omnimerge-v6-Q4_K_M.gguf`** (this repo), served on `llama.cpp` with MTP
speculative decoding enabled. The score below belongs to THIS quant — other
tiers in this repo were not run.


**v6 is first of ten in this cohort, and the safety result is the one worth reading.**


| model | quant | Total Points (mean of 5 seeds) | 95% CI | safety-critical failures (5 seeds) |
|---|---|---|---|---|
| **Qwen3.8-27B-Omnimerge-v6** | Q4_K_M | **156.4 ±3.5** | [152.0, 160.8] | **3** |
| Qwen3.8-27B (base) | UD-Q4_K_M | 150.8 ±2.5 | [147.7, 153.9] | 9 |
| Ornith-1.5-35B | IQ4_XS | 146.2 ±2.6 | [143.0, 149.4] | 10 |
| Qwen3.6-27B-Omnimerge-v4 | Q4_K_M | 146.2 ±2.7 | [142.9, 149.5] | 16 |
| Qwen3.6-27B | Q4_K_M | 144.0 ±3.4 | [139.8, 148.2] | 14 |
| Qwen3.6-35B-A3B | IQ4_XS | 141.6 ±2.4 | [138.6, 144.6] | 15 |

v6 scores **+10.2 pts over v4** and **+5.6 over its own Qwen3.8 base**, with
non-overlapping CIs against v4.

**The safety number is not a rounding difference.** Every other model in the cohort —
including both bases — fails **TC-60 (Cross-Turn Sleeper Injection) on 5 of 5 seeds**.
v6 never fails it. Its 3 failures across five runs are 2× TC-58 (Fake System Message in
File) and 1× TC-33 (Hallucination Resistance); it has no TC-31 or TC-34 failure at all,
where v4 fails both on every seed. `Safety & Boundaries` 23.2/26 (89.2%) vs v4's
18.4/26 (70.8%).

Category profile (mean over 5 seeds): perfect on Tool Selection, Parameter Precision,
Restraint & Refusal, Localization, Instruction Following, Toolset Scale and Creative
Composition. Weakest at **Autonomous Planning 3.4/6 (56.7%)** — below v4's 4.0/6 — and
`Context & State` 14.8/20 (74.0%). Hard Mode 33.0/38 (86.8%) leads the cohort, but only narrowly — the Qwen3.8 base is 32.8/38 (86.3%). The wide gap is `Safety & Boundaries`: 89.2% here vs 78.5% for the next-best model.


![Tool-calling benchmark](toolbench_scores.png)

### Full cohort

| model | quant | Total Points (mean, 5 seeds) | 95% CI | safety-critical (5 seeds) |
|---|---|---|---|---|
| Qwen3.8-27B-Omnimerge-v6 | Q4_K_M | 156.4 ±3.5 | [152.0, 160.8] | 3 |
| Qwen3.8-27B (base) | UD-Q4_K_M | 150.8 ±2.5 | [147.7, 153.9] | 9 |
| Ornith-1.5-35B | IQ4_XS | 146.2 ±2.6 | [143.0, 149.4] | 10 |
| Qwen3.6-27B-Omnimerge-v4 | Q4_K_M | 146.2 ±2.7 | [142.9, 149.5] | 16 |
| Qwen3.6-27B (base) | Q4_K_M | 144.0 ±3.4 | [139.8, 148.2] | 14 |
| Qwen3.6-35B-A3B (base) | IQ4_XS | 141.6 ±2.4 | [138.6, 144.6] | 15 |
| Qwen3.6-27B-A3B-CoderX | Q4_K_M | 137.4 ±4.9 | [131.3, 143.5] | 17 |
| Ornith-1.5-27B-A3B-Coder | IQ4_XS | 136.8 ±4.8 | [130.9, 142.7] | 12 |
| Ornith-1.5-27B-A3B-CoderX | IQ4_XS | 134.0 ±2.5 * | [130.8, 137.2] | 14 |
| Qwen3.6-27B-A3B-Coder | Q4_K_M | 123.2 ±2.3 | [120.4, 126.0] | 15 |

`*` one seed (s42) is graded on 174 pts, not 176 — see that model's card.

<details>
<summary><b>Basis — read before comparing these numbers to anything</b></summary>

- **Scorer: `tool-eval-bench` v2.6.0** (the pip/uv-installed package, verified via
  `tool_eval_bench.__file__`, not a git checkout). An earlier note in the runner claimed
  `cf54b4b` (v2.6.0-45); that is **wrong and has been corrected** — no cell ever ran it.
  All 50 cells ran the same v2.6.0, so the cohort is internally consistent.
- **v2.6.0 carries a known scorer crash on TC-62.** `email_calls[-1]` raises `IndexError`
  when a model sent no valid CFO email; the orchestrator catches it and returns
  **FAIL / 0 points while keeping the scenario in the denominator**. It hits 11 of 38
  scored cells, 2 pts each, and it is **not neutral — it concentrates on the weakest
  models**. Later harness commits credit that behaviour instead, so a fixed scorer would
  **raise** affected scores, unevenly.
- 5 paired seeds [42–46], 64k context, context-pressure 0.25, max 8 turns, 120 s timeout,
  thinking enabled, sampler **temp 0.6 / top-p 0.95 / top-k 20** (not greedy).
- Served on `llama.cpp b1788384120-c588c4f47` with MTP speculative decoding enabled
  (`nextn=YES spec=mtp`), one model per GPU, sequential.
- **Quant tiers are not uniform across the cohort** (Q4_K_M for the Omnimerge/A3B rows,
  IQ4_XS for Ornith and 35B-A3B, UD-Q4_K_M for the Qwen3.8 base). Cross-row gaps
  therefore carry a quantisation component and are not purely architectural.
- Do **not** pool these with the r/LocalLLaMA published tool-eval-bench figures: those
  were run at 256k context and are a different basis despite the shared scorer version.

</details>

