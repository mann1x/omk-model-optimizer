# omk_qad — model-agnostic 2-bit QAD (IQ2 family) → GGUF

Repo overview, recipes, datasets, evaluations and per-model results: [`.github/README.md`](../.github/README.md).

omk additions on top of NVIDIA Model-Optimizer (this repo is a private mirror of `NVIDIA/Model-Optimizer`;
`upstream` remote = NVIDIA, push disabled). Everything omk lives in `omk_qad/`; upstream files are untouched,
so rebases on `upstream/main` stay clean.

Reference recipe: NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB (experts IQ2_XXS + QAD, Jaccard loss, LR 1e-5).

| Module | What it does |
|---|---|
| `losses.py` | Jaccard distillation loss (full vocab, and top-K + rest for a cached teacher), KL control. `L = 2·TV/(1+TV)`. |
| `census.py` | Classifies every parameter by structure (fused experts, router, shared/dense MLP, attention, linear attention, embeddings), applies a format policy with block-size fallbacks, proves 100 % parameter coverage, emits the ModelOpt quant config. |
| `ptq.py` | PTQ of any HF causal LM (text tower) to the census policy → ModelOpt GGML-packed HF checkpoint. The PTQ floor. |
| `gguf_export.py` | Packed checkpoint → GGUF through llama.cpp's own converter, carrying the packed bytes verbatim; verifies every quantized tensor with gguf-py's reference decoders (which match libggml exactly). |
| `tiny.py` | Tiny random models with the exact architecture of the targets (qwen3_5_moe_text / qwen3_5_text; `dense_tied` = Qwen3.5-4B config, tied embeddings) for CPU tests. |
| `layout.py` | Per-GGUF-tensor layout decided by llama.cpp itself: the converter is run on index probes (source id / row / column, no weights read) to get each GGUF tensor's source rows and column order, and `llama-quantize --dry-run <ftype>` on a sparse shape-only GGUF gives the plan (default IQ2_M with experts / dense MLP fixed at IQ2_XXS). |
| `ggml_lib.py` | libggml's own encoder/decoder via ctypes (what llama-quantize runs) for every type ModelOpt lacks (IQ3_S, K-quants, ...). |
| `fakequant.py` | Training-time fake quant from the layout: latents (fp16, stochastic rounding) served with each GGUF segment encoded in GGUF block order (ModelOpt for IQ1/IQ2/Q8_0, libggml otherwise); forward hooks swap served weights in one module at a time; the gradient is applied inside backward (momentum-free Adafactor, per expert). |
| `train.py` | QAD driver: `--phase cache` (teacher top-K once), `train` (lossless gate, Jaccard top-K+rest, Bolt LR schedule), `export`. |
| `qad_export.py` | Latents -> GGUF at the plan with the encoders training served; proves every quantized tensor is the latent rows/columns training encoded, re-reads and verifies every tensor and type. Writes the imatrix next to the GGUF (`<stem>.imatrix.gguf` + `.imatrix.pt`). |
| `mtp.py` | MTP (nextn) head of Qwen3.5/3.6 checkpoints rebuilt from the trunk's own layer class (HF ForCausalLM drops `mtp.*`), loaded from the checkpoint, run as llama.cpp's `graph_mtp`; exported as `blk.<n_layer>.nextn.*` + `nextn_predict_layers`. |
| `imatrix.py` | Importance matrix collected from the teacher during `--phase cache` (per expert for fused experts, MTP head included), saved as `imatrix.pt` and llama.cpp's `imatrix.gguf`; weights every encoder in training and export. |
| `ternary.py` | PrismML Bonsai 2 ternary codec (plan T1): `top:K` (PrismML: 86, ties kept) and exact `opt` (imatrix-weighted L2) ternarizers with fp16 scales; bit-exact PQ2_0 (type 142) / PTQ1_0 (type 143) pack/unpack, checked against the C reference and PrismML's own 27B file; raw GGUF reader for these types; `HadamardBasis` (Bonsai 2 contract, seeded signs). In the trainer a ternary-typed weight is held folded, H(s·W) in GGUF column order, and served as unfold(ternary) (gradient = fold of the served gradient). Ternary latents are weighted by `imatrix.folded.pt`, E[(H s x)²] in GGUF column order, collected in the same teacher pass. Ternary types are planned per class (`--fixed dense_mlp=pq2_0`, `--override attention=pq2_0`), through an IQ2_XXS placeholder in llama-quantize's dry run, or by `layout.retype`. The model has ONE Hadamard block (`layout['hadamard_block']`, the largest of 1024/512/256 dividing every folded input width), because the runtime reads a single `prism.hadamard.block_size`. Export writes types 142/143 with `prism.hadamard.*` (v2 tied / v1 untied, explicit seeded signs) and ships the training payloads byte for byte. These files run only on PrismML's fork and opencoti's llamafile, not on stock llama.cpp or ollama. |
| `ternary_gate.py` | Engine-gate controls of a ternary GGUF: `unfold` (ternary tensors unfolded to F32, no prism keys: the weights the ternary file serves, as a plain model) and `flip` (signs negated: a negative control). An engine serves what training served if its KLD on the ternary file against the unfold control is ~0, and the flip control is far. Tiny tied model, both engines: 0.00075 vs 0.52. |
| `multi.py` | Separate students over one shared teacher (e.g. an IQ2_XXS and a ternary student). A spec (YAML/JSON) lists the students, each with its own layout, hyper-parameters and GPU. Shared once: the layout probe, the teacher top-K cache, the eval rows and the imatrix pass. Per student: one `omk_qad.train` process each, run concurrently; each pauses, resumes and fails on its own. Phases: `plan` (one layout per student; an existing one is never replaced; ternary block == the shared folded imatrix's), `cache`, `preflight` (each student alone for a few steps: `PEAK_MEM`; students sharing a GPU must fit, or `train` refuses), `train`, `export`, `report` (one table, per eval family). |
| `recovery.py` | bf16 recovery mode (`train --mode bf16`): the model's own bf16 weights in `--train-scope` train against the cache's teacher (any model of the same tokenizer), no quantizer, no layout; latents updated in backward with stochastic-rounding bf16 writes (no fp32 master). Scope `gemma4` = the reviewed Gemma-4 MoE scope (attention `q/k/v/o` + per-layer dense `mlp`; count derived, router opt-in via `--train-router`, experts / embedding / vision never). Final-logit soft capping (Gemma-4: 30) in teacher cache, eval and loss. The lossless gate runs only when the cache teacher is the student. Export (`--out-hf`) writes the source HF checkpoint back with only the trained tensors replaced, every other tensor verified bit-identical. |
| `rollouts.py` | Agentic conversations -> `.roles.jsonl` windows with a role per token: 0 ignore (prompt, tool output), 1 teacher (reasoning, prose, tool-call structure, the stop token), 2 anchor (code arguments, fenced code: KL to the original student, `train --anchor` cache pass), 3 loop (teacher + unlikelihood, `--ul-weight`). Rendered with the model's chat template, split into segments at later user messages (Gemma-4 drops earlier reasoning), every scored turn's prompt checked to be a prefix of its render; long segments become stride-overlapping windows scoring each token once. |

Environment: the `omk-modelopt` conda env, built and verified by `bash omk_qad/conda_env/setup_conda_env.sh`
(pinned requirements, this repo editable, CUDA 12.x toolkit rules, `OMK_ENV_OK` check). See
[`conda_env/README.md`](conda_env/README.md).

```bash
ENV=$(conda info --base)/envs/omk-modelopt/bin/python
$ENV -m omk_qad.census --model DIR                       # meta-device census, no weights read
$ENV -m omk_qad.ptq --model DIR --out PACKED [--policy routed_experts=iq2_s,...] [--algorithm imatrix --calib-text F]
$ENV -m omk_qad.gguf_export --src PACKED --out X.gguf [--llama-cpp /opt/llama.cpp] [--decoded-f32]
CUDA_VISIBLE_DEVICES= $ENV -m pytest -o addopts="" omk_qad/tests/   # CPU suite; artifacts in $OMK_QAD_WORK
```

Census on the real targets (text tower, default Bolt policy):

| Model | 2-bit class | Total |
|---|---|---|
| Qwen3.6-27B-A3B-Coder | routed experts 23.15 B → 5.97 GB | **8.48 GB** |
| Qwen3.6-27B dense | dense MLP 17.1 B → 4.41 GB | 14.83 GB (linear-attn + attention at Q8_0 dominate; needs a lower format there) |

QAD (one GPU; the teacher is cached, never resident while training):

```bash
$ENV -m omk_qad.layout --model TEXT_CFG_DIR --out L.pt [--rest IQ2_M] [--compare IQ2_M,IQ3_S,Q3_K_S,bolt]
$ENV -m omk_qad.train --phase cache  --model M --layout L.pt --cache C --data a.jsonl=3,b.parquet=1 --eval-data e.txt
$ENV -m omk_qad.train --phase train  --model M --layout L.pt --cache C --out O                 # stage 1
$ENV -m omk_qad.train --phase train  --model M --layout L.pt --cache C --out O --stage 2 --branch NAME [...]
$ENV -m omk_qad.train --phase export --model M --layout L.pt --load O/stage1/final.pt --gguf X.gguf   # no --load = PTQ floor
```

**Layout.** The fixed classes (type set by hand) default to routed/shared experts + dense MLP at IQ2_XXS
(`--fixed`). Fixed or not, every quantized tensor trains at its planned type (except `train --export-only` classes).
Everything else follows a llama.cpp preset's rules (`--rest`: IQ2_M default, IQ2_S, IQ2_XS, IQ3_XXS, IQ3_XS,
IQ3_S, IQ3_M, Q2_K(_S), Q3_K_S/M, Q4_K_S/M, or `bolt` = Q8_0). `--override class=type` and
`--override-regex 'pattern=type'` refine it, in the precedence fixed > class > regex > preset. A layout with
no fixed class (pure llama.cpp IQ2_M) is refused. The MTP head defaults to Q4_K (`--mtp-type`, `rest`, or
`--no-mtp`).

**Runs.** `--phase cache` also collects the imatrix (first `--imatrix-seqs` training sequences, default 256);
training and export use it (`--imatrix auto` = the cache's), and export refuses a different one than training
used. Each run dir holds `last.pt` (rolling; written every `--save-every` 100 steps or `--save-minutes` 20,
on pause and at the end) and `final.pt` when the stage completes. Pause with Ctrl-C / SIGTERM or
`touch <run>/PAUSE`; relaunching the same command resumes from `last.pt`, and the result is bitwise identical
to an uninterrupted run (CPU test). Stage 1 lives in `O/stage1`. Each stage 2 is `O/stage2/<branch>`,
initialised weights-only from `O/stage1/final.pt` (`--init-from`), so several stage-2 recipes can start from
one stage 1. A finished stage or branch is immutable (a relaunch is refused).

Rest presets (llama.cpp dry-run sizes, `layout --compare`, MTP head at Q4_K included), 2-bit class fixed at IQ2_XXS:

| rest | IQ2_M | IQ2_S | IQ3_XXS | IQ3_S | IQ3_M | Q2_K | Q3_K_S | Q3_K_M | bolt (Q8_0) |
|---|---|---|---|---|---|---|---|---|---|
| A3B-Coder (GB) | **7.18** | 7.16 | 7.21 | 7.32 | 7.33 | 7.20 | 7.25 | 7.34 | 8.51 |
| 27B dense (GB) | **8.85** | 8.72 | 9.04 | 9.47 | 9.53 | 8.92 | 9.12 | 9.55 | 14.81 |

IQ2_M breakdown:

| Model | IQ2_XXS class | rest (llama.cpp IQ2_M rules) | Total |
|---|---|---|---|
| Qwen3.6-27B-A3B-Coder | routed + shared experts 5.99 GB | lm_head Q5_K, linear-attn Q4_K/IQ2_S, embed IQ3_S, attn IQ2_S/IQ3_S | **7.18 GB** |
| Qwen3.6-27B dense | MLP 4.41 GB | linear-attn Q4_K/IQ2_S, lm_head Q5_K, embed IQ3_S, attn IQ2_S/IQ3_S/Q4_K | **8.85 GB** |

Tied embeddings (e.g. Qwen3.5-4B): the one tensor is classed `lm_head`, trained fake-quantized at the head's planned type (llama.cpp ships it as `token_embd` with the output rule), imatrix from the head's input; the gradient sums the lookup and head uses (CPU test). The ModelOpt PTQ path (`ptq.py`) is not yet tested on a tied model.

Known gaps: the GGML CUDA extension is not yet built/tested here (CPU path only); the IQ2 re-encode cost per
step on GPU is unmeasured (CPU reference encoder: ~1.3 M params/s; libggml K-quants run on CPU, `--ggml-refresh N`);
the token embedding is quantized at export only; on-policy teacher rollouts are not generated yet (the cache
takes any text/chat data); the MTP head's draft acceptance in llama.cpp is wired (draft-mtp engages on the
exported tiny model) but its numerical match to `graph_mtp` is only provable on a real head (GPU smoke).
