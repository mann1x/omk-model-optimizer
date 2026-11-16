# K-FAC rounding — curvature-aware GGUF quantization, and seeding a QAD from it

**What it does.** Re-rounds the MLP weights of a `llama-quantize` GGUF so that the quantization error is small
*where the model is sensitive*, not where the weights are large. The output is a GGUF of exactly the same layout,
size and types as the `llama-quantize` tier it starts from — only the payload bytes of the MLP tensors change — so it
runs on stock llama.cpp / ollama, and it carries the same metadata (imatrix provenance keys included). Measured on
Gemma-4 A4B models with a 44-row top-256 KL against the F16 model, it removes 6 % (Q6_K) to 24 % (IQ4_XS) of the
quantization KL at identical file size, and 17 % at IQ2_XXS. Its error-fed latents are also the **seed of a QAD run**:
the trainer starts from the K-FAC rounding instead of the RTN rounding, which is where the IQ2_XXS tiers of the two
published Gemma-4 agentic models come from.

Used for: [ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF)
and [ManniX-ITA/gemma-4-20B-v9-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF)
(every tier except IQ2_XXS, which is the seeded QAD). Model folders:
[`models/gemma-4-26b-a4b`](../models/gemma-4-26b-a4b/README.md), [`models/gemma-4-20b-v9-agentic`](../models/gemma-4-20b-v9-agentic/README.md).

## Why a curvature

Around its BF16 weights W0 a distillation student equals its teacher, so the loss gradient is ~0 and for any
quantization Δ = Q(W) − W0 the loss is the quadratic `L ≈ ½ Δᵀ H Δ`. llama-quantize minimises `Σ imatrix[col] · Δ²`
per row: a diagonal, input-side only approximation of H. K-FAC keeps the whole Kronecker-factored curvature per
linear layer, `H ≈ G ⊗ A`, with `A = E[x xᵀ]` over the layer's inputs and `G = E[g gᵀ]` over the loss gradients at
the layer's output; the loss of a rounding is then `½ tr(Δ A Δᵀ G)` per layer and expert, and no forward pass is
needed to evaluate or improve a rounding. `G` is the **true Fisher**: at each scored position one label is sampled
from the model's own distribution and `−log p(y)` is back-propagated. The maths and its probes are in the module
docstrings of [`omk_qad/kfac.py`](../../omk_qad/kfac.py), [`kfac_validate.py`](../../omk_qad/kfac_validate.py),
[`kfac_probe.py`](../../omk_qad/kfac_probe.py) and [`kfac_round.py`](../../omk_qad/kfac_round.py) (bm-895).

Two findings decided the collection recipe (2026-11-15 probes, 26B):

- attention-sink and "void" positions carry gradient norms 50–4,000× the median and inflate the early layers'
  curvature ~1000×; neither a residual-norm cut (`--drop-resid`) nor Rademacher signing (`--rademacher`) removes them.
  **`--drop-grow 10`** does: an absolute cut on the gate_up-output gradient norm of the rows entering G (every legitimate
  position in every window of every layer stays below ~25; the void positions are 28–1,040).
- the per-layer KL has a Δ-independent floor from downstream route flips; the prediction is a ranking tool, not an
  absolute KL. The rounding is therefore always **scored on the GGUF** (`step0_gguf_student_eval`), never trusted from
  the quadratic alone.

## Pipeline

| step | command | output |
|---|---|---|
| 1. factors | `omk_qad.kfac` | `factors_w512_s2048_g10/layer_NN.pt`: A and G per layer (routed experts per expert, dense MLP) |
| 2. PTQ reference | `llama-quantize --imatrix … [attention rule] F16 → TIER` | the tier exactly as llama.cpp makes it (the floor and the output's template) |
| 3. rounding | `omk_qad.kfac_round --stems all --qtype auto --ref ptq.gguf` | `kfac.gguf`: same layout and size, MLP payloads re-rounded; `--save-latents` for a QAD seed |
| 4. twin score | `step0_gguf_student_eval` on both files | `ptq.eval.json`, `kfac.eval.json`: top-256 KL vs the model's own F16 on the held-out rows |
| 5. (QAD) | `omk_qad.train --init-latents <latents>` | a QAD run that starts from the K-FAC rounding ([qad.md](qad.md)) |

### 1. Factors

```bash
PY=$(conda info --base)/envs/omk-modelopt/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m omk_qad.kfac --model MODEL_DIR --cache runs/<qad>/cache \
    --out runs/<model>_kfac/factors_w512_s2048_g10 --device cuda:0 \
    --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10
```

- `--cache` is an omk_qad teacher cache ([qad.md §2](qad.md)): its tokenised training windows are the inputs (the first
  512 rows of the seeded order, cropped to 2,048 tokens); the labels are sampled from the model under `--model` itself,
  so a cache of a *sibling* model with the same tokenizer is fine (the 20B v9-agentic used the 26B agentic cache).
- `--dense` also collects the dense MLP (gate/up share one input factor; three output factors); `--down-g diag` keeps a
  diagonal G for the routed down projection (its full G is 2560² per expert); `--layers-per-pass 3` bounds GPU memory
  (10 passes for 30 layers).
- Size and time on a 96 GB Blackwell: 26B agentic 77 GB, 20B v9-agentic 59 GB, ~75 min each. Factor files are never
  overwritten; the scripts refuse an existing `--out`.

### 2. PTQ reference — the Gemma-4 A4B attention rule

Every tier below Q4_K_M carries the attention projections at **Q6_K**, Q4_K_M and above at **Q8_0** (float tiers
none). This is the rule omnimergekit's `quantize_gguf.py` emits for Gemma-4 A4B and the published ladders follow it:

```bash
AT=q6_k   # tiers below Q4_K_M (IQ*, Q2_K, Q3_K_*, Q4_K_S); q8_0 for Q4_K_M, Q4_K_L, Q5_*, Q6_K, Q8_0
llama-quantize --imatrix IMATRIX --tensor-type "blk\.\d+\.attn_(q|k|v|output)\.weight=$AT" F16.gguf ptq.gguf TIER 8
```

The imatrix is **required** at every tier, K-FAC or not: it is the PTQ reference's input, the rounder's per-column
importance (see below) and the GGUF's `quantize.imatrix.*` provenance. Which imatrix: the 20B ladder uses the
AtomicChat corpus imatrix of the published model (`imatrix.dat`, 9,590 chunks, `--parse-special`); the 26B ladder
uses the QAD cache's own corpus imatrix (`cache/imatrix.gguf`, 258 chunks of 16,384 tokens from the training windows,
per expert), the same file its IQ2_XXS QAD trains and exports with.

### 3. Rounding

```bash
CUDA_VISIBLE_DEVICES=0 OMK_GGML_THREADS=8 CUDA_HOME=/usr/local/cuda-12.8 $PY -m omk_qad.kfac_round \
    --factors runs/<model>_kfac/factors_w512_s2048_g10 --base F16.gguf --imatrix IMATRIX \
    --qtype auto --stems all --ref ptq.gguf --out kfac.gguf --report kfac.json \
    --device cuda:0 --llama-cpp /path/to/llama.cpp          # [--save-latents DIR] for a QAD seed
```

| flag | meaning |
|---|---|
| `--base` | the BF16/F16 GGUF the factors were collected on (W0 of every tensor) |
| `--imatrix` | the llama.cpp imatrix the PTQ used: `E[x²]` per column (1 where an expert saw no token), the same weighting the encoder applies |
| `--qtype auto` | each tensor takes the type its `--ref` tensor carries, so a mixed tier (K-quant with Q6_K attention, IQ4_NL down experts…) rounds as laid out |
| `--stems all` | every MLP tensor: routed `ffn_gate_up_exps` + `ffn_down_exps`, dense `ffn_gate` / `ffn_up` / `ffn_down` (`gate_up` and `routed` are the smaller sets) |
| `--method gptq` (default) | GPTQ at the encoder's block granularity: one 256-column super-block of every row is quantized jointly with the real encoder, its error folded into the later columns through the Cholesky factor of the damped inverse of A (`--damp 0.01` = fraction of mean diag A); `rtn` reproduces llama-quantize's bytes exactly and is the floor |
| `--ste K` | K straight-through refinement steps on the full quadratic `G Δ A` after GPTQ (default 0: the published tiers use GPTQ alone) |
| `--llama-cpp` | the llama.cpp tree whose `libggml` encodes the K-quants, IQ1/IQ2_XS, IQ4_NL, Q8_0 and legacy types (the exact code llama-quantize runs, CPU, `OMK_GGML_THREADS`); IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS use the project's CUDA kernels, byte-identical to llama-quantize |
| `--save-latents DIR` | the error-fed fp16 latents of the routed gate_up per layer (`layer_NN.pt`): the QAD seed |

Behaviour worth knowing:

- `--out` is written **first as a byte copy of `--ref`**, then each MLP tensor's payload is replaced in place, layer
  by layer. A killed rounder therefore leaves a file that *looks* complete but is partly the PTQ: a `kfac.gguf` is
  valid only when its log ends with `KFACR_DONE`. Every script in this repo checks that sentinel before scoring,
  archiving or publishing (bug-949).
- For every tensor the `rtn` codes are compared byte for byte with `--ref` (`ENCODER_MATCH`); a mismatch
  (`KFACR_ENCODER_MISMATCH`) means the encoder is not llama-quantize's and the run is wrong. Expected count: 0.
- GPTQ rounds each block to the **latent dtype** (fp16) before encoding, so an fp16-latent trainer seeded from the
  saved latents starts at exactly the rounding that was scored (the seed gate below checks this to 0.0005 KL).
- Time: CPU-encoded K-quant tiers 60–80 min for a 30-layer model (20B Q6_K 4,907 s), kernel-encoded IQ tiers ~45 min.
  The rounder needs ~11 GB of GPU memory.

### 4. Twin score

Both files are scored with `step0_gguf_student_eval` (llama.cpp's own kernels on both sides, `--topk 256`): the
teacher is the model's unquantised GGUF, the rows are the held-out `.roles.jsonl` windows of the QAD cache (44 rows,
4 per family), scored on the assistant positions only (role > 0, anchor as teacher). `klk` = mean top-256 KL(teacher‖
student), `top1` = argmax agreement. A model with no QAD cache gets its reference set from its own F16 GGUF
(`build_eval_from_gguf.py` in the 20B folder: same rows, this model's top-64).

### 5. Seeding a QAD

`omk_qad.train --phase train … --init-latents <save-latents dir>` starts the routed gate_up latents from the K-FAC
rounding instead of RTN (only gate_up is seeded; the trainer re-encodes everything else from bf16). The seed is
exact only if every stage shares one imatrix (PTQ reference, rounder, trainer, export) and the trainer's layout types
match the reference's. The **seed gate** before training: export the untrained seeded student (`--phase export
--init-latents`, no checkpoint) and check (a) `SEED_LATENTS … 30 gate_up` in the export log, (b) attention bytes
identical to the K-FAC reference and routed IQ2_XXS bytes within 0.5 % (`gguf_tensor_diff`), (c) |Δ klk| ≤ 0.0005
against the K-FAC reference's score. The chains in the model folders run this gate and refuse to train on a failure.

## Results

44-row top-256 KL vs the model's own F16, `klk` / `top1`, identical bytes per pair (2026-11-16, llama.cpp b12386):

| model | tier | llama-quantize + imatrix | K-FAC | ΔKL |
|---|---|---:|---:|---:|
| 26B agentic | Q4_K_M | 0.00798 / 0.9796 | 0.00660 / 0.9815 | −17.3 % |
| 26B agentic | IQ4_XS | 0.01530 / 0.9715 | 0.01157 / 0.9751 | −24.4 % |
| 26B agentic | IQ2_XXS (arm E, gate_up only) | 0.06748 / 0.9392 | 0.05757 / 0.9469 | −14.7 % |
| 26B agentic | IQ2_XXS (arm E, all MLP) | 0.06748 / 0.9392 | 0.05586 / 0.9480 | −17.2 % |
| 20B v9-agentic | Q4_K_M | 0.00981 / 0.9753 | 0.00822 / 0.9781 | −16.1 % |
| 20B v9-agentic | Q6_K | 0.00311 / 0.9862 | 0.00292 / 0.9868 | −6.0 % |

The gain grows as the tier gets coarser. The complete per-tier tables are on the two GGUF model cards (regenerated
by the publisher as tiers land) and in the model folders.

Seeded QAD, 26B IQ2_XXS (layout E, attention Q6_K, `--init-latents` from the arm-E K-FAC rounding, 600 steps,
seq 4096, lr 2e-6, `--loss kl`): seed 0.05915 → step 300 0.04933 → step 400 0.04870 → step 500 0.04832 (top1 0.9494),
i.e. the trained IQ2_XXS removes another ~18 % of the KL on top of the rounding's 17 %, against the unseeded RTN start
of 0.0675.

## Reproduce

The scripts as run (bs2, one Blackwell per model) are in the model folders:
[`gemma-4-20b-v9-agentic/quant/`](../models/gemma-4-20b-v9-agentic/quant/) (`kfac_collect_20b.sh`,
`build_eval_from_gguf.py`, `kfac_ladder_twins.sh`, `kfac_ladder_lane.sh`, `kfac_publish_bs2.py`) and
[`gemma-4-26b-a4b/kfac/`](../models/gemma-4-26b-a4b/kfac/) (`kfac_collect_r.sh`, `step1_layouts.sh`,
`qad_seeded_E_chain.sh`). `kfac_ladder_twins.sh <26b|20b> <TIER> all` is one tier end to end: PTQ → rounding → twin
score. The 26B and 20B differ only in the base GGUF, imatrix, factors and eval reference set at the top of that script.
