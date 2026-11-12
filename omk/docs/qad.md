# Mode 1 — QAD: quantization-aware distillation to a GGUF

**What it does.** Trains a *quantized* student against its own full-precision teacher. Every weight that the
final GGUF stores quantized is held as an fp16 latent and served, in every forward pass, fake-quantized in the exact
GGUF type and block order llama.cpp will read (IQ1/IQ2/IQ3, K-quants, Q8_0). The loss is a distillation loss on the
teacher's cached top-K logits. The output is a GGUF that runs on stock llama.cpp / ollama, with every tensor proven to
be the rows training encoded.

Use it when a model must fit a smaller GPU at 2–3 bits and the stock `llama-quantize` tier at that size loses too
much (the Qwen3.5-4B IQ2_XXS student: HumanEval 38.4 vs 10.4 for the stock IQ2_XXS of the same F16).

`--mode qad` is the trainer default. The ternary formats (PQ2_0 / PTQ1_0) are the same mode with ternary types in
the layout: see [ternary.md](ternary.md). Training a model's own bf16 weights with no quantizer is
[bf16-recovery.md](bf16-recovery.md).

## Pipeline

| step | command | output |
|---|---|---|
| 1. layout | `omk_qad.layout` | `layout.pt`: every GGUF tensor's source rows, column order and planned type |
| 2. teacher cache | `omk_qad.train --phase cache` | top-K logits of every training sequence, eval rows, imatrix |
| 3. train | `omk_qad.train --phase train` | `<out>/stage1/{last,final,ckpt_stepNNNNNNN}.pt` |
| 4. export | `omk_qad.train --phase export` | the GGUF (+ `<stem>.imatrix.gguf`), verified tensor by tensor |

The same four steps run from one spec with `omk_qad.multi` (several students over one teacher cache: see
[qad-ternary-together.md](qad-ternary-together.md)). The single-student commands below are what `multi` runs for
each student.

## 1. Layout

```bash
ENV=$(conda info --base)/envs/omk-modelopt/bin/python
$ENV -m omk_qad.layout --model MODEL_DIR --out runs/x/layout.pt --rest IQ2_M      # experts / dense MLP fixed at IQ2_XXS
$ENV -m omk_qad.layout --model MODEL_DIR --out runs/x/layout.pt --compare IQ2_M,IQ3_S,Q3_K_M   # size table first
```

- The layout is decided by llama.cpp itself: its converter is run on index probes (no weights read), then
  `llama-quantize --dry-run <ftype>` on a shape-only GGUF gives each tensor's type.
- `--fixed class=type,...` sets classes by hand (default: routed/shared experts + dense MLP at `iq2_xxs`).
  `--rest` is the llama.cpp preset for everything else (`IQ2_M` default; `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ3_M`,
  `Q2_K`, `Q3_K_S/M`, `Q4_K_S/M`, or `bolt` = Q8_0). `--override class=type` and `--override-regex 'pat=type'` refine
  it (precedence fixed > class > regex > preset). A layout with no fixed class is refused.
- MTP (nextn) heads of Qwen3.5/3.6 are planned at Q4_K (`--mtp-type`, `--no-mtp`).

## 2. Teacher cache (the teacher runs once)

```bash
$ENV -m omk_qad.train --phase cache --model MODEL_DIR --layout runs/x/layout.pt --cache runs/x/cache \
    --data corpus_a.jsonl=3,corpus_b.parquet=1,calib.txt=1 --eval-data eval_code.jsonl,eval_chat.jsonl \
    --n-seq 16000 --n-eval 32 --seq 4096 --topk 64
```

- `--data` is a comma list of `path[=weight]` (`.txt`, `.jsonl` with `text` or `messages`, `.parquet`); chat rows
  are rendered with the model's chat template. `--eval-data` files are one eval family each (named by file stem).
- The cache also collects the **imatrix** (first `--imatrix-seqs` sequences, default 256; per expert for fused
  experts; the folded statistics ternary needs). Training and export use it (`--imatrix auto`), and export refuses
  an imatrix other than the one training used.
- A teacher too large for torch on one GPU can run from a GGUF: `--teacher-gguf F.gguf` (libllama via
  `omk_qad/llama_topk`; build line in `omk_qad/gguf_teacher.py`). That path has no imatrix.

## 3. Train

```bash
$ENV -m omk_qad.train --phase train --model MODEL_DIR --layout runs/x/layout.pt --cache runs/x/cache --out runs/x \
    --loss jaccard --lr 1e-5 --steps 2400 --bs 8 --warmup 50 --eval-every 300
```

| flag | default | notes |
|---|---|---|
| `--loss` | `jaccard` | `jaccard` (Bolt: `2·TV/(1+TV)` on top-K + rest), `kl` = KL(t‖s), `rkl` = KL(s‖t) |
| `--lr` / `--min-lr` / `--warmup` | 1e-5 / 1e-7 / 50 | Bolt schedule (warmup, then cosine to min) |
| `--bs`, `--accum` | 8, 1 | rows per micro-batch, micro-batches per update |
| `--latent-dtype` | fp16 | latents with stochastic rounding |
| `--ggml-refresh` | 1 | re-encode libggml-typed (K-quant, IQ3) segments every N steps (CPU encoders) |
| `--export-only` | `embedding` | classes quantized at export only |
| `--mtp` | `train` | distil the MTP head with the trunk detached, or `freeze` |
| `--guard-ratio` | 0 (off) | divergence guard: roll back + cut LR when the loss exceeds ratio × EMA |
| `--snapshot-every` / `--snapshot-keep` | 100 / 4 | numbered `ckpt_stepNNNNNNN.pt` besides the rolling `last.pt` |

Runs are **pausable and resumable bit for bit**: Ctrl-C / SIGTERM or `touch <out>/stage1/PAUSE` checkpoints at
the step boundary and exits; the same command then resumes from `last.pt`. `echo pause > <out>/stage1/control`
checkpoints and waits in-process until the file says `run`. A resume with different run-critical settings is
refused unless `--allow-change KEY` names them. Stage 2 variants start weights-only from stage 1:
`--stage 2 --branch NAME [--init-from <out>/stage1/final.pt]`. A finished stage or branch is immutable.

W&B logs when `WANDB_API_KEY` is set (`--wandb-project`, `--wandb-name`, `--no-wandb`). Every step logs loss, LR,
grad norm, update/weight ratio, tokens and per-source loss.

## 4. Export

```bash
$ENV -m omk_qad.train --phase export --model MODEL_DIR --layout runs/x/layout.pt --load runs/x/stage1/final.pt \
    --gguf out/model-omk-IQ2_XXS.gguf
# no --load = the PTQ floor of the same layout (the baseline QAD must beat)
```

Export proves every quantized tensor is the latent rows/columns training encoded, re-reads the file and checks every
tensor against gguf-py's reference decoders, and writes the imatrix next to it.

## Measuring

[`evaluation.md`](evaluation.md): KLD vs the F16 GGUF and HumanEval/HE+, next to stock `llama-quantize` tiers of the
same F16 and llama.cpp build. Worked example with numbers: [Qwen3.5-4B](../models/qwen3.5-4b/README.md).
