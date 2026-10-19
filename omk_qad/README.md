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

Known gaps: the GGML CUDA extension is not yet built/tested here (CPU path only); ForCausalLM loading drops
the MTP head, so exports carry none until it is grafted back from the base; the QAD trainer is not written yet.
