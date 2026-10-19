# omk_qad — model-agnostic 2-bit QAD (IQ2 family) → GGUF

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
| `tiny.py` | Tiny random models with the exact architecture of the targets (qwen3_5_moe_text / qwen3_5_text) for CPU tests. |
| `layout.py` | Per-GGUF-tensor layout decided by llama.cpp itself: the converter is run on index probes (source id / row / column, no weights read) to get each GGUF tensor's source rows and column order, and `llama-quantize --dry-run <ftype>` on a sparse shape-only GGUF gives the plan (default IQ2_M with experts / dense MLP fixed at IQ2_XXS). |
| `ggml_lib.py` | libggml's own encoder/decoder via ctypes (what llama-quantize runs) for every type ModelOpt lacks (IQ3_S, K-quants, ...). |
| `fakequant.py` | Training-time fake quant from the layout: latents (fp16, stochastic rounding) served with each GGUF segment encoded in GGUF block order (ModelOpt for IQ1/IQ2/Q8_0, libggml otherwise); forward hooks swap served weights in one module at a time; the gradient is applied inside backward (momentum-free Adafactor, per expert). |
| `train.py` | QAD driver: `--phase cache` (teacher top-K once), `train` (lossless gate, Jaccard top-K+rest, Bolt LR schedule), `export`. |
| `qad_export.py` | Latents -> GGUF at the plan with the encoders training served; proves every quantized tensor is the latent rows/columns training encoded, re-reads and verifies every tensor and type. |

```bash
ENV=/root/anaconda3/envs/omk-modelopt/bin/python
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
$ENV -m omk_qad.layout --model TEXT_CFG_DIR --out L.pt                  # config + tokenizer only; ~2.5 min CPU
$ENV -m omk_qad.train --phase cache  --model M --cache C --data a.jsonl=3,b.parquet=1 --eval-data e.txt
$ENV -m omk_qad.train --phase train  --model M --layout L.pt --cache C --out O
$ENV -m omk_qad.train --phase export --model M --layout L.pt --load O/state.pt --gguf X.gguf   # no --load = PTQ floor
```

IQ2_M layout (llama.cpp dry-run sizes), 2-bit class fixed at IQ2_XXS:

| Model | IQ2_XXS class | rest (llama.cpp IQ2_M rules) | Total |
|---|---|---|---|
| Qwen3.6-27B-A3B-Coder | routed + shared experts 5.99 GB | lm_head Q5_K, linear-attn Q4_K/IQ2_S, embed IQ3_S, attn IQ2_S/IQ3_S | **7.18 GB** |
| Qwen3.6-27B dense | MLP 4.41 GB | linear-attn Q4_K/IQ2_S, lm_head Q5_K, embed IQ3_S, attn IQ2_S/IQ3_S/Q4_K | **8.85 GB** |

Known gaps: the GGML CUDA extension is not yet built/tested here (CPU path only); ForCausalLM loading drops
the MTP head, so exports carry none until it is grafted back from the base; the IQ2 re-encode cost per step on GPU is unmeasured (CPU reference encoder: ~1.3 M params/s); the token
embedding is quantized at export only (IQ3_S encode is ~4 M params/s on CPU); no imatrix in the encoders yet;
on-policy teacher rollouts are not generated yet (the cache takes any text/chat data).
