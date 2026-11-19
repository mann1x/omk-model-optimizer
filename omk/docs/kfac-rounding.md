# K-FAC rounding — curvature-aware GGUF quantization, how to build a ladder, and seeding a QAD from it

**In one paragraph.** `llama-quantize` rounds each weight row so that `Σ imatrix[col] · Δ²` is small: a diagonal,
input-side-only picture of how much a rounding error hurts. K-FAC rounding keeps the *whole* per-layer curvature of
the model's loss around its full-precision weights, `H ≈ G ⊗ A`, and re-rounds the MLP tensors of a finished
`llama-quantize` GGUF against it, with the real llama.cpp encoders, so the output is a GGUF of **exactly the same
layout, types, size and metadata** as the stock tier — only the payload bytes of the MLP tensors differ — and it runs
on stock llama.cpp and ollama. Measured on the two published Gemma-4 agentic models with a held-out top-256 KL twin
test (same bytes, same rows, same kernels), it removes 6 % (Q8_0) to 41 % (Q3_K_M) of the quantization KL at identical
file size, 17 % at Q4_K_M and 24 % at IQ4_XS. The same machinery produces the **seed of a QAD run**: the 2-bit
IQ2_XXS tiers start from the K-FAC rounding instead of round-to-nearest and train from there.

Used for every tier of [ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF)
and [ManniX-ITA/gemma-4-20B-v9-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF)
(IQ2_XXS = the seeded QAD). Model folders with every script as run:
[`models/gemma-4-26b-a4b`](../models/gemma-4-26b-a4b/README.md) (`kfac/`),
[`models/gemma-4-20b-v9-agentic`](../models/gemma-4-20b-v9-agentic/README.md) (`quant/`).

Contents: [1. What K-FAC is](#1-what-k-fac-is) · [2. Where it comes from: OBS, OBQ, GPTQ](#2-where-it-comes-from-obs-obq-gptq-and-what-this-adds) ·
[3. How the pieces fit](#3-how-the-pieces-fit) · [4. How to build a K-FAC ladder](#4-how-to-build-a-k-fac-ladder-step-by-step) ·
[5. Seeding a QAD](#5-seeding-a-qad-the-iq2_xxs-tiers) · [6. Results](#6-results) · [7. Limits and what the probes taught](#7-limits-and-what-the-probes-taught) ·
[8. References](#8-references).

## 1. What K-FAC is

**K-FAC = Kronecker-Factored Approximate Curvature** (Martens & Grosse, 2015). It is an approximation of a neural
network's curvature matrix — the Fisher information matrix, which at a well-trained model equals the Gauss-Newton
Hessian of the loss — that is small enough to store and invert per layer.

For one linear layer `y = W x` (W is `out × in`) the exact Fisher of the `out·in` weights is a dense
`(out·in) × (out·in)` matrix: for a 2816 × 1408 expert that is 4 million × 4 million entries, impossible to hold.
K-FAC observes that the gradient of the loss with respect to W is an outer product, `∇_W L = g xᵀ` with `x` the layer
input and `g = ∂L/∂y` the gradient at the layer output, so the Fisher `E[vec(g xᵀ) vec(g xᵀ)ᵀ]` is approximated by the
**Kronecker product of two small second-moment matrices**:

```
H  ≈  G ⊗ A        A = E[x xᵀ]   (in × in,  the layer's input covariance)
                   G = E[g gᵀ]   (out × out, the covariance of the loss gradient at the layer output)
```

The approximation assumes the input and the output gradient are statistically independent; it is the standard
assumption behind K-FAC as an optimizer (natural-gradient preconditioning) and it holds well enough here, where it is
used not to train but to **measure the loss of a weight perturbation**. With Δ the perturbation of W (for us, the
rounding error `Q(W) − W`), the second-order loss increase is a quadratic form that K-FAC turns into two small
matrix products per layer:

```
L(Δ)  ≈  ½ vec(Δ)ᵀ H vec(Δ)  ≈  ½ tr(Δ A Δᵀ G)          gradient:  ∂L/∂Δ ≈ G Δ A
```

Why the loss is a pure quadratic: around its own full-precision weights W0 a distillation student equals its
teacher, so the first-order term (the gradient of KL(teacher ‖ student)) is zero and only the curvature remains. Any
quantization's damage is then `½ tr(Δ A Δᵀ G)`, evaluated without a forward pass, for any format, from factors that
are collected **once**.

Which "G" matters. The empirical gradient of the distillation loss itself is ~0 at W0, so it carries no curvature.
`omk_qad.kfac` collects the **true Fisher**: at every scored position one token `y` is sampled from the model's own
next-token distribution and `−log p(y)` is back-propagated. The expectation of `g gᵀ` under the model's own sampling
is exactly the Fisher, which equals the Hessian of KL(model ‖ perturbed model) at Δ = 0 — the quantity a quantization
should minimise. `A` is the plain second moment of the layer input, which for a routed expert is accumulated only
over the tokens the router sent to that expert (so each expert has its own `A_e`, `G_e` and token count `n_e`).

## 2. Where it comes from: OBS, OBQ, GPTQ, and what this adds

The lineage is the "optimal brain" family of second-order weight-removal methods:

| method | year | what it does | curvature it uses |
|---|---|---|---|
| OBD (LeCun, Denker, Solla) | 1990 | prune the weight whose removal raises the loss least, `½ H_ii w_i²` | diagonal Hessian |
| OBS (Hassibi, Stork) | 1993 | the same with the full Hessian, **and compensates the remaining weights** by `−w_i H⁻¹ e_i / [H⁻¹]_ii` | full Hessian + inverse |
| OBQ (Frantar, Alistarh) | 2022 | OBS applied to quantization, one weight at a time, with the *layer-wise reconstruction* loss `‖W X − Ŵ X‖²` whose Hessian is `H = 2 X Xᵀ` — the input second moment, our `A` | `A` only, exact, per row |
| **GPTQ** (Frantar, Ashkboos, Hoefler, Alistarh) | 2022 | makes OBQ tractable for LLMs: quantize the columns in a fixed order (not greedily), update all rows at once, propagate each column's rounding error into the not-yet-quantized columns through the **Cholesky factor of the damped inverse Hessian** `(A + λI)⁻¹` | `A` only, shared by all rows |
| `llama-quantize --imatrix` | 2024 | round-to-nearest per block, scale search weighted by `imatrix[col] = E[x_col²]` | the **diagonal** of `A` |
| **K-FAC rounding (this)** | 2025 | GPTQ's error feedback **at the GGUF encoder's own block granularity with the real llama.cpp encoders**, driven by the collected `A_e` per expert; the output-side factor `G_e` makes the quadratic the KL curvature, used to predict and validate every rounding and, optionally, to refine it | `G ⊗ A` per layer and expert |

What GPTQ taught us and what had to change for GGUF. GPTQ rounds one column at a time because its codebook is a
per-channel or per-group scale: a column's codes are independent of its neighbours'. GGUF types are not like that:
an IQ4_XS, Q4_K or IQ2_XXS block encodes 32 or 256 consecutive weights **jointly** — one super-block scale, per-block
sub-scales (or a shared codebook for the IQ types) and the codes are chosen by a search over the whole block. The
rounder therefore applies GPTQ's idea at the block's granularity: every 256-column super-block of **every row** of the
tensor is quantized together with the exact encoder llama-quantize would run (`omk_qad.iq3` / `omk_qad.iq4` CUDA
kernels and libggml itself for the K-quants, IQ2 family, IQ4_NL, Q8_0 — all byte-identical to `llama-quantize`, checked
on every tensor), its rounding error is folded into the later columns through the Cholesky factor of the damped
inverse of `A_e` (`--damp 0.01`, eigenvalue floor `1e-6 · λmax`), and the next block is encoded on the compensated
weights. When a whole column block is fixed at once the OBS compensation is the same for any `G`, so at the shipped
tiers the rounding decision is GPTQ on the full `A_e`; `G_e` enters through `kfac_validate` / `kfac_probe` (predicting
and ranking the KL of a rounding) and through the optional `--ste K` straight-through refinement of the latent on the
full gradient `G Δ A` (default 0 on the published ladders).

Three things make this different from "GPTQ for GGUF":

- **the encoders are llama.cpp's**, bit for bit (`rtn` mode reproduces the `llama-quantize` payload byte for byte,
  the `ENCODER_MATCH` gate), so a K-FAC file is a legal GGUF of the stock type and the comparison with the stock tier
  is a rounding-only difference;
- **the curvature is per expert**: a routed expert's `A_e` is the second moment of the tokens *it* saw, with `n_e` as
  its weight in the whole-model loss (an expert that never fired has zero factors and is left at the stock rounding);
- **the input imatrix is still used**, as `E[x²]` per column inside the encoders' scale search (the same weighting
  `llama-quantize` applies) and as the GGUF's `quantize.imatrix.*` provenance keys; K-FAC adds the factors on top, it
  does not replace the imatrix.

## 3. How the pieces fit

```
BF16/F16 base GGUF ──┬─► llama-quantize --imatrix (+ attention rule) ──► ptq.gguf  (the stock tier: floor + template)
                     │                                                        │
HF model + windows ──┴─► omk_qad.kfac  ──► factors/layer_NN.pt (A_e, G_e, n_e) │
                                                      │                       ▼
                                 omk_qad.kfac_round --stems all --qtype auto --ref ptq.gguf ──► kfac.gguf (same byte count)
                                                      │                       │
                                                      └─ --save-latents ──► QAD seed (IQ2_XXS)
                                                                              ▼
                              step0_gguf_student_eval on ptq.gguf and kfac.gguf ──► twin score (klk, top1) ──► publish the winner
```

| module / script | role |
|---|---|
| [`omk_qad/kfac.py`](../../omk_qad/kfac.py) | collects `A_e`, `G_e`, `n_e` per layer and expert (+ dense MLP) on cached windows; `--check` first proves the instrumented model reproduces the logits |
| [`omk_qad/kfac_validate.py`](../../omk_qad/kfac_validate.py) | predicts the KL of any GGUF arm from its Δ alone (`Σ_e n_e/N · ½ tr(Δ_e A_e Δ_eᵀ G_e)` + dense), with the diagonal-Fisher and plain `Σ Δ²` baselines; the ranking gate |
| [`omk_qad/kfac_probe.py`](../../omk_qad/kfac_probe.py) | measures one layer's true KL against the quadratics on live windows (Δ-scale scan, per-position diagnostics); how the collection recipe was fixed |
| [`omk_qad/kfac_round.py`](../../omk_qad/kfac_round.py) | the rounder: `rtn` (= llama-quantize, the floor), `gptq` (block-wise error feedback on `A_e`), `--ste K` (refinement on `G Δ A`); writes `--out` as a byte copy of `--ref` with the MLP payloads replaced |
| `step0_gguf_student_eval.py` (`runs/g4_agentic_iq2_r16k/`, bs2) | the twin score: top-256 KL and top-1 of a GGUF against the reference set, llama.cpp kernels on both sides (`llama_topk`) |
| [`quant/build_eval_from_gguf.py`](../models/gemma-4-20b-v9-agentic/quant/build_eval_from_gguf.py) | an eval reference set for a model without a QAD cache: its own F16 GGUF's top-K on the shared rows |
| [`quant/kfac_ladder_twins.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_ladder_twins.sh) | **one tier end to end**: PTQ → rounding → twin score, sentinels `LT_PTQ / LT_KFAC / LT_TWIN / LT_DONE` |
| [`quant/kfac_ladder_lane.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_ladder_lane.sh), [`kfac_lanes_after_cache.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_lanes_after_cache.sh) | a lane = a list of tiers for one model run back to back, disk-gated |
| [`quant/kfac_ladder_archive.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_ladder_archive.sh), [`kfac_publish_bs2.py`](../models/gemma-4-20b-v9-agentic/quant/kfac_publish_bs2.py), [`kfac_ollama_bs2.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_ollama_bs2.sh) | sha-verified archive to the backup host, HF upload with card rendering, ollama tags |
| [`kfac/qad_seeded_E_chain.sh`](../models/gemma-4-26b-a4b/kfac/qad_seeded_E_chain.sh), [`qad/qad20_seeded_E_chain.sh`](../models/gemma-4-20b-v9-agentic/qad/qad20_seeded_E_chain.sh) | the IQ2_XXS chain: layout → PTQ reference → K-FAC seed → seed gate → QAD → export |

## 4. How to build a K-FAC ladder, step by step

The commands below are the ones the two published ladders ran (bs2, one RTX PRO 6000 Blackwell per model, llama.cpp
**b12386**). Paths are the scripts' defaults; every script refuses to overwrite an existing output.

### 4.0 Prerequisites

- the `omk-modelopt` conda env (`bash omk_qad/conda_env/setup_conda_env.sh`, ends with `OMK_ENV_OK`), CUDA 12.8
  toolkit for the IQ3/IQ4 CUDA encoders (`CUDA_HOME=/usr/local/cuda-12.8`; a 13.x nvcc cannot build the torch
  extensions);
- a built llama.cpp checkout (`llama-quantize`, `libggml`) — **the same build for every step of a ladder**;
- `llama_topk` built against that llama.cpp (`omk_qad/llama_topk/`), the top-K scorer the twin test uses;
- the model as an HF checkpoint (for the factors) **and** as its unquantised GGUF (BF16 for a bf16 checkpoint): the
  GGUF is `W0` of every tensor the rounder re-rounds, so it must be the export of the checkpoint the factors were
  collected on (`kfac_validate --hf` compares layer 0 bit for bit);
- an imatrix for that GGUF (any `llama-imatrix` output; the 20B used the AtomicChat corpus imatrix with
  `--parse-special`, the 26B its QAD cache's own corpus imatrix);
- a set of tokenised windows with role masks (an `omk_qad` teacher cache, [qad.md](qad.md)): the factors are
  collected on its training windows, the twin score on its held-out rows. A sibling model with the same tokenizer can
  lend its cache (the 20B v9-agentic used the 26B agentic cache: the labels are sampled from the model under
  `--model`, so only the token streams are borrowed);
- disk: factors 59–77 GB per model, each tier ~2 × its GGUF size while the twin exists, plus the archive.

### 4.1 Collect the factors (once per model, 75 min – 2¼ h, 59–77 GB)

```bash
PY=$(conda info --base)/envs/omk-modelopt/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m omk_qad.kfac --model MODEL_DIR --cache runs/<qad>/cache \
    --out runs/<model>_kfac/factors_w512_s2048_g10 --device cuda:0 \
    --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10
```

- `--windows 512 --seq 2048`: the first 512 windows of the cache's seeded order, cropped to their densest 2,048
  tokens (4,096 does not fit next to the model); 614,086 scored positions on the 26B.
- `--layers-per-pass 3`: fp32 accumulators are ~5.4 GB per layer (128 experts, hidden 2816, moe_inter 704), so the
  30 layers are collected in 10 passes over the windows.
- `--down-g diag`: the routed down projection keeps a diagonal `G` (its full `G` would be 2816² per expert);
  `--dense`: the dense MLP too (gate and up share one input factor, three output factors).
- `--drop-grow 10`: drop the rows whose gate_up-output gradient norm exceeds 10 from `G` — the attention-sink /
  "void" positions (§7). It removes 0.6–0.9 % of the rows in layers 0–3 and almost nothing after layer 13.
- `--check` runs first by itself: the instrumented expert loop must reproduce the original logits on one window and
  every gradient hook must fire; `KFAC_CHECK OK` then `KFAC_DONE` in the log.
- Output: `layer_NN.pt` per decoder layer with `gu_A[e]`, `gu_G[e]`, `dn_A[e]`, `dn_G[e]`, `n[e]` and the dense
  factors. The files are never overwritten; collect a second recipe into a new directory.

As run: [`kfac/kfac_collect_r.sh 0 g10`](../models/gemma-4-26b-a4b/kfac/kfac_collect_r.sh) (26B),
[`quant/kfac_collect_20b.sh`](../models/gemma-4-20b-v9-agentic/quant/kfac_collect_20b.sh) (20B).

### 4.2 The eval reference set (once per model)

The twin score needs the model's **own** full-precision top-K at every scored position of the held-out rows, so that
the KL measures the quantization alone:

- a model with a QAD cache has it already: `cache/eval.pt` (26B: 44 rows, 4 per family, assistant positions only,
  256,853 scored positions, the bf16 teacher's top-64 computed in torch);
- a model without one gets it from its own GGUF with llama.cpp's kernels:

```bash
$PY quant/build_eval_from_gguf.py --gguf MODEL-F16.gguf --rows <sibling>/cache/eval.pt --out runs/<model>_kfac/eval.pt \
    --bin llama_topk --omk /path/to/omk-model-optimizer --topk 64      # EVAL_FROM_GGUF OK <scored>
```

The two reference styles put the two models' `klk` on different scales (torch bf16 teacher vs the model's own F16
GGUF): compare tiers **within** a model, never a 26B `klk` with a 20B one.

### 4.3 One tier: PTQ reference → rounding → twin score

```bash
bash quant/kfac_ladder_twins.sh 26b Q4_K_M all        # or 20b; stages: ptq | kfac | score | all
```

What the script does, and what you would type by hand:

**a. PTQ reference — the stock tier, with the Gemma-4 A4B attention rule.** Every tier below Q4_K_M carries the
attention projections at Q6_K, Q4_K_M and above at Q8_0 (the rule `quantize_gguf.py` emits for this family; the MoE
is sensitive to attention and the projections are small):

```bash
AT=q8_0   # Q4_K_M, Q4_K_L, Q5_*, Q6_K, Q8_0;   q6_k for IQ*, Q2_K, Q3_K_*, Q4_K_S;   none for F16/BF16
llama-quantize --imatrix IMATRIX --tensor-type "blk\.\d+\.attn_(q|k|v|output)\.weight=$AT" BASE.gguf ptq.gguf TIER 8
```

This file is both the **floor** (what llama.cpp alone produces, scored in every table as *llama-quantize + imatrix*)
and the **template**: the K-FAC file is a byte copy of it with the MLP payloads replaced. Run `--dry-run` first; the
script logs the `falling back` count (tensors whose shape forced another type).

**b. Rounding.**

```bash
CUDA_VISIBLE_DEVICES=0 OMK_GGML_THREADS=8 CUDA_HOME=/usr/local/cuda-12.8 $PY -m omk_qad.kfac_round \
    --factors runs/<model>_kfac/factors_w512_s2048_g10 --base BASE.gguf --imatrix IMATRIX \
    --qtype auto --stems all --ref ptq.gguf --out kfac.gguf --report kfac.json \
    --device cuda:0 --llama-cpp /path/to/llama.cpp                 # [--save-latents DIR] for a QAD seed
```

| flag | meaning |
|---|---|
| `--base` | the unquantised GGUF the factors were collected on: `W0` of every tensor |
| `--imatrix` | the imatrix the PTQ used: `E[x²]` per column (1 where an expert saw no token), the encoders' scale-search weighting |
| `--qtype auto` | each tensor takes the type its `--ref` tensor carries, so a mixed tier (Q6_K attention, IQ4_NL down experts, …) rounds as laid out |
| `--stems all` | every MLP tensor: routed `ffn_gate_up_exps` + `ffn_down_exps`, dense `ffn_gate` / `ffn_up` / `ffn_down` (`gate_up`, `routed`, `down_dense` are the subsets) |
| `--method gptq` (default) | block-wise GPTQ on `A_e` (§2); `--damp 0.01` |
| `--ste K` | K straight-through steps on the full quadratic `G Δ A` after GPTQ, best iterate kept (default 0 = the published ladders) |
| `--llama-cpp` | the tree whose `libggml` encodes the K-quants, IQ2_XS/S/M, IQ4_NL, Q8_0 and legacy types on the CPU (`OMK_GGML_THREADS`); IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS use the project's CUDA kernels — all byte-identical to llama-quantize |
| `--save-latents DIR` | the error-fed fp16 latents of the routed gate_up per layer (`layer_NN.pt`): the QAD seed (§5) |
| `--report` | per layer and stem: the kfac / diagonal / plain-`Σ Δ²` predictions for `rtn` and for the final rounding |

Behaviour that matters operationally:

- `--out` is written **first as a byte copy of `--ref`**, then each MLP tensor is replaced in place, layer by layer.
  A killed rounder leaves a file that *looks* complete but is partly the stock tier: a `kfac.gguf` is valid **only when
  its log ends with `KFACR_DONE`**. Every script here checks that sentinel before scoring, archiving or publishing.
- For every tensor the `rtn` codes are compared byte for byte with `--ref`: `ENCODER_MATCH`, expected mismatches 0. A
  `KFACR_ENCODER_MISMATCH` means the encoder is not llama-quantize's and the file must not ship.
- Experts with `n_e = 0` in the factors (a few in layers 0–5 of the 26B) are left at the stock rounding.
- Time on a 30-layer model: K-quant tiers (CPU encoders, 8 threads) 60–80 min (20B Q6_K 4,907 s); IQ tiers on the
  CUDA kernels ~45 min; GPU memory ~11 GB. A finished tier has `kfac.gguf`, `kfac.gguf.sha256`, `kfac.json`, `kfac.log`.

**c. Twin score.** Both files on the same reference set, llama.cpp kernels on both sides:

```bash
$PY step0_gguf_student_eval.py --gguf kfac.gguf --eval runs/<model>_kfac/eval.pt --bin llama_topk --omk OMK --topk 256 \
    --device cuda:0 --out kfac.eval.json      # and the same for ptq.gguf;  GGUF_STUDENT_EVAL OK
```

`klk` = mean top-256 KL(reference ‖ quant) per scored position, `top1` = argmax agreement, plus the per-family
breakdown (agentic, opencode, swe, cp, kimi31b, hermes, math, science, ifchat ×2, multilingual). The script prints one
`LT_TWIN <TIER> size … llama.cpp <klk/top1> K-FAC <klk/top1>` line into `ladder/twins.log`. **Publish the K-FAC file
only if it beats its twin**; the twin is kept until both scores exist, then removed on the owner's grant (`--rm-ptq`
or the `.rm_ptq_granted` marker).

### 4.4 The whole ladder

The published ladder is the 15 `llama-quantize` tiers Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, IQ4_XS, IQ4_NL,
Q3_K_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ2_M, IQ2_S, IQ2_XS, plus IQ2_XXS from the seeded QAD (§5). Tiers are independent,
so they run as **lanes** — lists of tiers for one model, each lane one `kfac_ladder_twins.sh … all` after the other,
pausing when `/mnt/sdc` falls below a free-space floor (`MIN_FREE_G`) and skipping a tier whose `kfac.eval.json` and
`kfac.gguf.sha256` already exist:

```bash
bash quant/kfac_ladder_lane.sh 20b "Q5_K_S Q4_K_S IQ4_NL Q3_K_M IQ3_M IQ3_XS IQ2_M IQ2_S IQ2_XS Q8_0"    # lane B20
bash quant/kfac_ladder_lane.sh 26b "…"                                                                   # lane C26
```

Two lanes plus a driver for the four dual tiers (IQ4_XS, IQ3_XXS, Q5_K_M, Q6_K × both models) built the 32 tiers in
about 36 hours on one machine while a QAD trained on the second GPU. Three rules from running it:

- a lane's "done" test must be the **score + sha**, not the presence of `kfac.gguf` — the publisher reclaims the GGUF
  after archive + hub + ollama, and a lane that tested the file re-quantized 22 published tiers (bug-964);
- the stock twin is removed by the **last line of the score stage**; a rounder killed between `LT_PTQ` and `LT_TWIN`
  strands it, so the lane removes orphan twins of scored tiers on its next pass;
- a dormant "start after X" launcher is a loaded gun: arm it with a stale-marker (`.lac_disarm`, a 6 h `LAC_STALE`).

### 4.5 Archive, publish, tags

- **Archive first** (`kfac_ladder_archive.sh`, on the backup host): every scored tier is pulled with its `.sha256`,
  `.eval.json`, `kfac.json` and logs and verified; the tier gets an `.archived_solidpc` marker.
- **HF** (`kfac_publish_bs2.py`, a 48 h loop on the build host): for every tier with `KFACR_DONE` + twin score it
  uploads `<repo-name-minus-GGUF>-<TIER>.gguf` (republish = overwrite of the same name), checks the hub LFS sha against
  the local file, writes the `.gguf.sha256` sidecar, uploads the imatrix once per repo, and **renders the model card**
  from a hub manifest (`<!-- OMK:TIERS -->`: size, bpw, sha, stock vs K-FAC klk/top1, ΔKL; `<!-- OMK:STALE -->`:
  previous-build files; a bench table from `cards/<model>_bench.json`). A local copy is reclaimed only when the hub sha,
  the archive marker **and** the ollama push all exist (`KP_RM_HOLD` otherwise).
- **Ollama** (`kfac_ollama_bs2.sh`): each published tier as `text`, `vision-`, `mtp-` and `mtp-vision-` tags (`:latest`
  = Q4_K_M), the public recipe read back from the registry (renderer, parser, sampler), positive push markers.
- The HF token travels in a 0600 curl config, never on an argv; filenames follow the repo name
  (`gemma-4-26B-A4B-agentic-Q4_K_M.gguf`), IQ tier names are the llama.cpp names (IQ2_XXS, IQ3_XXS, IQ4_XS).

## 5. Seeding a QAD: the IQ2_XXS tiers

At 2 bits a rounding is not enough ([§7](#7-limits-and-what-the-probes-taught)); the IQ2_XXS tiers are trained students
([qad.md](qad.md)) that **start from the K-FAC rounding** of their exact layout instead of round-to-nearest.

1. **Layout.** The 26B arm ladder A–E ([`kfac/step1_layouts.sh`](../models/gemma-4-26b-a4b/kfac/step1_layouts.sh))
   fixed layout **E**: routed gate_up IQ2_XXS, routed down IQ4_NL, dense gate/up Q4_K, dense down IQ4_NL, embedding
   Q5_K; attention Q8_0 for the arm study, **Q6_K** (the family rule below Q4_K_M) for the shipped file = layout **E6**,
   ~9.5 GiB.
2. **PTQ reference** of that layout: `llama-quantize --imatrix … --token-embedding-type q5_K --tensor-type <per class>`
   (`E6.gguf`), scored.
3. **K-FAC seed**: `kfac_round --qtype IQ2_XXS --method gptq --ref E6.gguf --save-latents round/E_latents`
   (26B: 2,650 s, latents 29 GB; the down experts + dense MLP rounded in a second pass with `--stems down_dense`,
   [`kfac/kfac_down_dense_chain.sh`](../models/gemma-4-26b-a4b/kfac/kfac_down_dense_chain.sh)).
4. **Seed gate** (the chains refuse to train on a failure): export the untrained seeded student
   (`omk_qad.train --phase export --init-latents round/E_latents`, no checkpoint) and check (a) `SEED_LATENTS … 30
   gate_up` in the log, (b) attention bytes identical to the K-FAC reference and routed IQ2_XXS bytes within 0.5 %
   (`gguf_tensor_diff.py`; the 20B relaxed attention to ≤ 5e-4 of bytes for f16-subnormal weights, bug-957),
   (c) |Δ klk| ≤ 0.0005 against the reference's twin score. The seed is exact only if **one imatrix** is shared by PTQ
   reference, rounder, trainer and export, and the trainer's layout types match the reference's.
5. **QAD**: `omk_qad.train --phase train … --init-latents round/E_latents` — only the routed gate_up is seeded, the
   trainer re-encodes everything else from bf16. The published recipe: `--seq 4096 --bs 1 --steps 600 --lr 2e-6
   --min-lr 1e-7 --warmup 50 --beta1 0 --beta2 0.999 --clip 1.0 --loss kl --anchor-as-teacher --mtp train
   --mtp-weight 0.3 --refresh 4 --ggml-refresh 4 --latent-dtype fp16 --export-only embedding --topk 64`, the
   quantizer in the loop at every step, ~39 s/step on one Blackwell (26B), 35 s/step (20B).
6. **Export** with the same encoders (`qadE6_kfacseed_s600.gguf`), twin-scored against the layout's PTQ reference,
   published as `…-IQ2_XXS.gguf`.

Chains as run: [`kfac/qad_seeded_E_chain.sh`](../models/gemma-4-26b-a4b/kfac/qad_seeded_E_chain.sh) (26B, resumed
to 600 by [`qad_seeded_E_chain_resume600.sh`](../models/gemma-4-26b-a4b/kfac/qad_seeded_E_chain_resume600.sh)),
[`qad/qad20_seeded_E_chain.sh`](../models/gemma-4-20b-v9-agentic/qad/qad20_seeded_E_chain.sh) (20B: cache → layout →
PTQ reference → seed → gate → train → export, disk-gated).

## 6. Results

### 6.1 The 26B agentic ladder, tier by tier (the worked example)

Model: [ManniX-ITA/gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic) (Gemma-4
26B-A4B-it, 128 experts, 30 layers, BF16 recovery SFT against agentic loops). Base GGUF = the sft11 final BF16 export;
imatrix = the QAD cache's corpus imatrix (258 chunks × 16,384 tokens, per expert); factors `w512_s2048_g10` (77 GB,
2 h 14 min); reference set = the cache's 44 held-out rows, 256,853 assistant positions, bf16 teacher top-64 (torch).
`klk` = mean top-256 KL(teacher ‖ quant), `top1` = argmax agreement; stock and K-FAC files have identical byte counts.

| tier | size (GiB) | llama-quantize + imatrix klk / top1 | **K-FAC** klk / top1 | ΔKL |
|---|---:|---:|---:|---:|
| Q8_0 | 25.02 | 0.00236 / 0.9893 | **0.00223 / 0.9895** | −5.5 % |
| Q6_K | 21.33 | 0.00280 / 0.9881 | **0.00259 / 0.9887** | −7.5 % |
| Q5_K_M | 18.20 | 0.00419 / 0.9855 | **0.00376 / 0.9863** | −10.3 % |
| Q5_K_S | 17.14 | 0.00456 / 0.9847 | **0.00408 / 0.9858** | −10.5 % |
| Q4_K_M | 16.14 | 0.00798 / 0.9796 | **0.00660 / 0.9815** | −17.3 % |
| Q4_K_S | 14.67 | 0.01019 / 0.9766 | **0.00882 / 0.9788** | −13.5 % |
| IQ4_XS | 13.26 | 0.01530 / 0.9715 | **0.01157 / 0.9751** | −24.4 % |
| IQ4_NL | 13.71 | 0.01495 / 0.9717 | **0.01159 / 0.9753** | −22.5 % |
| Q3_K_M | 12.71 | 0.03656 / 0.9564 | **0.02167 / 0.9664** | −40.7 % |
| IQ3_M | 11.88 | 0.02476 / 0.9634 | **0.02001 / 0.9676** | −19.2 % |
| IQ3_XS | 11.27 | 0.03531 / 0.9556 | **0.02641 / 0.9630** | −25.2 % |
| IQ3_XXS | 11.02 | 0.04100 / 0.9542 | **0.02825 / 0.9615** | −31.1 % |
| IQ2_M | 10.11 | 0.07626 / 0.9358 | **0.05413 / 0.9479** | −29.0 % |
| IQ2_S | 9.65 | 0.09346 / 0.9282 | **0.07069 / 0.9405** | −24.4 % |
| IQ2_XS | 9.65 | 0.09309 / 0.9285 | **0.07082 / 0.9406** | −23.9 % |
| IQ2_XXS (layout E6) | 9.5 | 0.0675 / 0.9392 (RTN) | **0.0485 / 0.9496** (seeded QAD, step 600) | −28.2 % |

How to read it:

- **The gain grows as the tier gets coarser**, from 5 % at 8 bits to 24–41 % at 3–4 bits and ~25–30 % at 2 bits:
  the coarser the grid, the more there is to gain from *where* the error lands. Q3_K_M is the outlier (−41 %): its
  3-bit K-quant blocks have the most room for error feedback.
- **K-FAC moves a tier one rung up the ladder at no size cost.** K-FAC IQ4_XS (13.26 GiB, 0.01157) is within a few
  per cent of stock Q4_K_S (14.67 GiB, 0.01019); K-FAC Q3_K_M (12.71 GiB, 0.0217) beats stock IQ3_M
  (11.88 GiB, 0.0248) and stock IQ3_XS/IQ3_XXS by a wide margin; K-FAC IQ2_M (0.0541) is close to stock IQ3_XXS
  (0.0410) from 0.9 GiB less.
- **top-1 moves with the KL**: +0.2 points at Q4_K_M, +0.7 at IQ3_XXS, +1.2 at IQ2_M, i.e. the rounding is not only
  reshaping the tail of the distribution.

Per family (klk, stock / K-FAC) — the agentic and code families gain the most, which is what the factors were
collected on:

| tier | agentic | opencode | hermes (tool use) | kimi31b (agent tasks) |
|---|---:|---:|---:|---:|
| Q6_K | 0.0040 / 0.0034 | 0.0026 / 0.0022 | 0.0015 / 0.0012 | 0.0026 / 0.0025 |
| Q4_K_M | 0.0088 / 0.0064 | 0.0083 / 0.0056 | 0.0042 / 0.0028 | 0.0072 / 0.0050 |
| IQ4_XS | 0.0149 / 0.0100 | 0.0138 / 0.0090 | 0.0077 / 0.0051 | 0.0126 / 0.0087 |
| Q3_K_M | 0.0355 / 0.0168 | 0.0267 / 0.0148 | 0.0196 / 0.0102 | 0.0284 / 0.0156 |
| IQ3_XXS | 0.0367 / 0.0215 | 0.0364 / 0.0225 | 0.0230 / 0.0116 | 0.0321 / 0.0184 |
| IQ2_M | 0.0678 / 0.0398 | 0.0735 / 0.0379 | 0.0434 / 0.0206 | 0.0631 / 0.0374 |

The full per-family numbers for every tier are in each tier's `kfac.eval.json` / `ptq.eval.json` (archived with the
ladder) and on the GGUF card.

**The IQ2_XXS path on the 26B** (layout E6, same bytes in every row):

| arm | klk | top1 |
|---|---:|---:|
| stock `llama-quantize` + imatrix (RTN), layout E (attention Q8_0) | 0.0675 | 0.9392 |
| K-FAC rounding, gate_up only (GPTQ, 2,650 s) | 0.0576 | 0.9469 |
| K-FAC rounding, all MLP tensors (+ down experts, dense MLP; the seed) | 0.0559 | 0.9480 |
| the same seed re-laid at E6 (attention Q6_K, the shipped layout) | 0.0592 | 0.9460 |
| seeded QAD step 300 / 400 / 500 | 0.0493 / 0.0487 / 0.0483 | — / 0.9492 / 0.9494 |
| **seeded QAD step 600 (published IQ2_XXS)** | **0.0485** | **0.9496** |
| unseeded QAD of the same cache (qadC_kl7, 1,500 steps), for reference | 0.0678 → 0.062 (plateau) | — |

The rounding removes 17 % of the RTN error; 600 QAD steps from that seed remove another 18 %; a QAD that starts from
RTN needs its 1,500 steps to get to where the seed already is.

**Other evaluations of the 26B agentic.** The behavioural evidence for this model is the head-to-head on the
`manic_miner` agentic repair task (automaticum, Cerebriline SDK 4.100.277, 20 runs each, 30-minute budget): base
gemma-4-26B-A4B-it 3 / 20 runs finished, the agentic model **12 / 20** (Fisher p = 0.0079), reasoning-loop cuts per
run 8.0 → 1.0 (median). That test ran on the **BF16 GGUF**; the quantized tiers are gated by the KL twins above,
not by a separate agentic run per tier — the twin test is designed exactly so that a tier can be judged in two minutes
on the model's own assistant positions. Full analysis, per-run data and scripts:
[`models/gemma-4-26b-a4b/sft11/eval/`](../models/gemma-4-26b-a4b/sft11/eval/README.md). The K-FAC Q4_K_M and the
seeded IQ2_XXS are also the files behind the owner's hands-on ollama tags on the 16 GB workstation.

### 6.2 The 20B v9-agentic ladder (same recipe, own F16 reference)

[ManniX-ITA/gemma-4-20B-v9-agentic](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic): the 98-expert prune of
the 26B agentic retrained with sft12. Factors 59 GB collected on the 26B cache's windows (same tokenizer; labels
sampled from the 20B); imatrix = AtomicChat corpus imatrix (9,590 chunks, `--parse-special`); reference set = the
same 44 rows scored by the model's own F16 GGUF top-64 (`build_eval_from_gguf.py`), so the `klk` scale differs from
the 26B's.

| tier | size (GiB) | llama-quantize + imatrix | **K-FAC** | ΔKL |
|---|---:|---:|---:|---:|
| Q8_0 | 19.71 | 0.00266 / 0.9881 | **0.00247 / 0.9881** | −7.1 % |
| Q6_K | 16.83 | 0.00311 / 0.9862 | **0.00292 / 0.9868** | −6.0 % |
| Q5_K_M | 14.41 | 0.00485 / 0.9831 | **0.00455 / 0.9835** | −6.0 % |
| Q5_K_S | 13.60 | 0.00543 / 0.9820 | **0.00496 / 0.9831** | −8.7 % |
| Q4_K_M | 12.83 | 0.00981 / 0.9753 | **0.00822 / 0.9781** | −16.1 % |
| Q4_K_S | 11.63 | 0.01236 / 0.9720 | **0.01058 / 0.9745** | −14.4 % |
| IQ4_XS | 10.55 | 0.01766 / 0.9670 | **0.01419 / 0.9707** | −19.7 % |
| IQ4_NL | 10.90 | 0.01767 / 0.9670 | **0.01412 / 0.9706** | −20.1 % |
| Q3_K_M | 10.13 | 0.04007 / 0.9503 | **0.02542 / 0.9616** | −36.6 % |
| IQ3_M | 9.48 | 0.02969 / 0.9579 | **0.02508 / 0.9620** | −15.5 % |
| IQ3_XS | 9.01 | 0.03995 / 0.9493 | **0.03091 / 0.9575** | −22.6 % |
| IQ3_XXS | 8.80 | 0.04650 / 0.9478 | **0.03283 / 0.9562** | −29.4 % |
| IQ2_M | 8.10 | 0.08867 / 0.9265 | **0.06083 / 0.9408** | −31.4 % |
| IQ2_S | 7.75 | 0.10640 / 0.9192 | **0.07841 / 0.9320** | −26.3 % |
| IQ2_XS | 7.75 | 0.10595 / 0.9190 | **0.07845 / 0.9324** | −26.0 % |
| IQ2_XXS (E6) | 7.5 | 0.0790 / 0.9306 (RTN) | **0.0564 / 0.9420** (seeded QAD, step 600; seed 0.0669 / 0.9381) | −28.7 % |

The only full benchmark suite run on a K-FAC file so far is this model's **K-FAC Q6_K** (canonical 9-bench suite,
llama.cpp b12386, greedy + presence 0.025, thinking on): GPQA-diamond 47.98, AIME 76.67, MATH500 83.00, GSM8K 94.00,
ARC-Challenge 87.03, IFEval 94.00, HumanEval 98.17, HumanEval+ 90.85, LCB-medium-55 94.55, LCB-medium-100 95.00,
LCB-hard-77 87.01, MultiPL-E 89.33; tool-eval-bench hardmode 142.6 / 176 on the **K-FAC Q4_K_M** (5 seeds, 95 % CI
141.2–144.0). Columns for the parent and the v7-coder sibling and the basis caveats are in
[`cards/20b_bench.json`](../models/gemma-4-20b-v9-agentic/cards/20b_bench.json). These numbers measure the model
through the quant; the quant's own quality is the twin KL above.

## 7. Limits and what the probes taught

- **A quadratic in Δ holds while Δ is small.** It is excellent at 4–8 bits, good at 3, and at 2 bits the rounding
  alone leaves a 0.05–0.07 KL that only training recovers — hence the seeded QAD for IQ2_XXS. K-FAC also treats layers
  as independent (no cross-layer error build-up), which is why the predicted scale is a ranking tool rather than an
  absolute KL.
- **The per-layer KL has a Δ-independent floor** from downstream route flips (a scale-0 control is exactly 0, a
  scale-0.1 Δ already pays the floor). Validation uses the quadratic part, KL(Δ) − floor, and every rounding is
  **scored on the GGUF**, never trusted from the surrogate alone.
- **Attention sinks and "void" positions.** One backward of the *summed* loss gives a window-start, delimiter or
  turn/channel token the coherent sum of all later losses' gradients, 50–4,000× the median, and the K-FAC-expand
  `E[g gᵀ]` squares that sum: the early layers' curvature came out ~1,000× too large (layer 0 predicted 15.0 nats vs
  0.014 measured). A residual-norm cut did not find them (their residual norms are ordinary), random-sign
  (Rademacher) per-loss signing did not change them (the gradients are the positions' own); what did was an
  **absolute cut on the gate_up-output gradient norm**, `--drop-grow 10` (every legitimate position in every window of
  every layer stays below ~25; the void positions are 28–1,040). After it the whole-model prediction is within 2.5–8×
  of the measurement (ranking 4/4 arms correct), deep layers within 1.5×, early layers ~5× over on a few hot windows.
- **Only same-attention arms are comparable** in the surrogate: attention and embeddings carry no factors. The
  ladders keep attention at the family rule and re-round only the MLP.
- **Experts that saw no token** in the collection windows (2/2/1/1 in layers 0/1/2/5 of the 26B) have zero factors
  and keep the stock rounding.
- **What is not yet done**: `--ste` refinement on `G Δ A` is implemented but unused on the published ladders; a
  cross-layer bit budget from the per-layer quadratics needs the per-layer calibration scalar the early layers lack;
  attention and embeddings are not re-rounded.

## 8. References

- J. Martens, R. Grosse, *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*, ICML 2015
  (K-FAC: `F ≈ E[g gᵀ] ⊗ E[a aᵀ]` per layer).
- Y. LeCun, J. Denker, S. Solla, *Optimal Brain Damage*, NeurIPS 1990. B. Hassibi, D. Stork, *Second order derivatives
  for network pruning: Optimal Brain Surgeon*, NeurIPS 1993.
- E. Frantar, D. Alistarh, *Optimal Brain Compression: a framework for accurate post-training quantization and
  pruning*, NeurIPS 2022 (OBQ).
- E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers*, ICLR 2023 (column-ordered OBQ with lazy updates and the Cholesky inverse-Hessian).
- llama.cpp `llama-quantize` / `llama-imatrix` (the importance matrix: per-column `E[x²]` weights for the block scale
  search), and the GGUF block formats this work encodes into.
- Related 2025 work that also brings the output-side curvature into PTQ: YAQA, GuidedQuant (end-to-end Fisher /
  Hessian guidance for layer-wise quantization). This implementation differs in rounding at the GGUF block granularity
  with llama.cpp's own encoders and in collecting per-expert factors of a MoE.
