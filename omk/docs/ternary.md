# Mode 2 — Ternary recovery: QAD to PrismML ternary (PQ2_0 / PTQ1_0)

**What it does.** The same trainer as [QAD](qad.md), with the chosen classes planned at the ternary types of
PrismML's Bonsai 2 format: one trit {−1, 0, +1} per weight and one fp16 scale per group of 128, packed as **PQ2_0**
(GGUF type 142, 2.125 bpw) or **PTQ1_0** (type 143, 1.75 bpw; same trits, denser packing).
A plain ternarization of a model (every weight → {−1, 0, +1} × scale) destroys it (Qwen3.5-4B: KL 8.0 at init); the
distillation *recovers* it against the model's own teacher (KL 0.46 after 2,400 steps, HumanEval 17.7).

How a ternary weight trains:

- it is held **folded**, `H(s·W)` in GGUF column order (`H` = the Bonsai 2 Hadamard basis with seeded signs), and
  served as `unfold(ternary(folded))`; the gradient is the fold of the served gradient;
- the ternarizer is `--ternary-method top:86` (PrismML's: the 86 largest |w| of each 128-weight group, ties kept,
  scale = mean of the kept |w|) or `opt` (the exact imatrix-weighted L2 optimum per group);
- latents are weighted by the folded imatrix `E[(H s x)²]`, collected in the same teacher pass as the normal one;
- the model has **one** Hadamard block (`layout['hadamard_block']`: the largest of 1024 / 512 / 256 dividing every
  folded input width), because the runtime reads a single `prism.hadamard.block_size`.

**Runtime.** Ternary GGUFs run only on PrismML's llama.cpp fork and opencoti's llamafile build, **not** on stock
llama.cpp or ollama. Measuring them needs that engine's `perplexity` build ([install.md](install.md)).

## Layout: which classes go ternary

Ternary types are planned per class, through an IQ2_XXS placeholder in llama-quantize's dry run:

```bash
ENV=$(conda info --base)/envs/omk-modelopt/bin/python
# dense MLP + attention + linear attention + (tied) head ternary; the rest by the IQ2_M preset
$ENV -m omk_qad.layout --model MODEL_DIR --out runs/t/layout.pt --rest IQ2_M \
    --fixed dense_mlp=pq2_0 --override attention=pq2_0,linear_attn=pq2_0,lm_head=pq2_0
```

For an MoE model put the experts at `pq2_0` with `--fixed routed_experts=pq2_0,shared_experts=pq2_0`. Classes left
out of the ternary set train at their llama.cpp type, exactly as in QAD.

## Train and export

```bash
$ENV -m omk_qad.train --phase cache  --model MODEL_DIR --layout runs/t/layout.pt --cache runs/t/cache \
    --data corpus.jsonl --eval-data eval.jsonl --n-seq 19500 --seq 1024
$ENV -m omk_qad.train --phase train  --model MODEL_DIR --layout runs/t/layout.pt --cache runs/t/cache --out runs/t \
    --loss kl --lr 6e-5 --steps 2400 --bs 8 --warmup 50 --ternary-method top:86
$ENV -m omk_qad.train --phase export --model MODEL_DIR --layout runs/t/layout.pt --load runs/t/stage1/final.pt \
    --gguf out/model-omk-PQ2_0.gguf --ternary-method top:86
```

What differs from an IQ2 student, from the Qwen3.5-4B run:

| | IQ2_XXS student | ternary student |
|---|---|---|
| loss | `jaccard` | `kl` |
| peak LR | 1e-5 | **6e-5** (ternary starts far from the teacher: init KL 8.0 vs 1.2) |
| export | stock GGUF types | types 142/143 + `prism.hadamard.*` keys (v2 tied / v1 untied, explicit seeded signs); the training payloads are shipped byte for byte |

Pass the same `--ternary-method` to export as to training (`omk_qad.multi` does it for you).

## Engine gate: does the runtime serve what training served?

```bash
$ENV -m omk_qad.ternary_gate unfold out/model-omk-PQ2_0.gguf out/model-unfold-F32.gguf   # the served weights as a plain model
$ENV -m omk_qad.ternary_gate flip   out/model-omk-PQ2_0.gguf out/model-flip.gguf         # signs negated: negative control
```

Run the ternary engine's KLD of the ternary file against the `unfold` control: ~0 means the engine computes what
training computed. The `flip` control must be far (tiny tied model, both engines: 0.00075 vs 0.52). Do this before
trusting any score from a new engine build.

## Running it next to an IQ2 student

A ternary and an IQ2 student of the same model share the layout probe, the teacher cache, the eval rows and the
imatrix pass. Train them together from one spec, on one GPU or on two: [qad-ternary-together.md](qad-ternary-together.md).
