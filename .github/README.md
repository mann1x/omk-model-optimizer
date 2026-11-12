# omk-model-optimizer

**Train models for llama.cpp: quantization-aware distillation to GGUF, ternary recovery, and BF16 recovery
training.** `omk_qad` is one trainer with three modes. In QAD it trains a quantized student against its own
full-precision teacher, with every weight fake-quantized in the exact format and block order the GGUF will hold
(IQ1/IQ2/IQ3, K-quants, Q8_0), then writes that GGUF and verifies every tensor. With ternary types in the layout it
recovers a PrismML ternary (PQ2_0 / PTQ1_0) model. In BF16 mode it trains a model's own bf16 weights, by distillation
from a bigger teacher or by SFT on role-masked agentic traces, and writes an HF checkpoint back. Model-agnostic: the
layout comes from llama.cpp's own converter and `llama-quantize` rules, not from per-model code.

This is the omk fork of [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer): upstream's README is the
root [`README.md`](../README.md), untouched. Everything omk adds lives in `omk_qad/` and `omk/`.

## The three modes

| mode | what it trains | output | runs on | guide |
|---|---|---|---|---|
| **QAD** (`--mode qad`, default) | fp16 latents served fake-quantized at their GGUF type (IQ2_XXS, IQ3, K-quants, …); distillation on the teacher's cached top-K | GGUF, every tensor verified | stock llama.cpp, ollama | [`omk/docs/qad.md`](../omk/docs/qad.md) |
| **Ternary recovery** (QAD with `pq2_0` / `ptq1_0` classes) | folded Hadamard-basis latents served as PrismML ternary; KL distillation recovers the ternarized model | ternary GGUF (types 142/143 + `prism.hadamard.*`) | PrismML fork, opencoti llamafile | [`omk/docs/ternary.md`](../omk/docs/ternary.md) |
| **BF16 recovery** (`--mode bf16`) | the model's own bf16 weights in a chosen scope, no quantizer; distillation (any same-tokenizer teacher, GGUF teachers included) or SFT (`--loss ce`) | HF checkpoint, only trained tensors replaced | anything (then quantize) | [`omk/docs/bf16-recovery.md`](../omk/docs/bf16-recovery.md) |

QAD and ternary students of one model share one teacher cache and train side by side, on one GPU or several:
[`omk/docs/qad-ternary-together.md`](../omk/docs/qad-ternary-together.md).

| | |
|---|---|
| [Install](#install) | conda env, llama.cpp, CUDA toolkit |
| [How it works](#how-it-works) | the QAD / ternary pipeline in one screen |
| [Recipes](#recipes) | one spec per trained model |
| [Datasets](#datasets) | the corpus catalogue |
| [Evaluations](#evaluations) | protocol and results |
| [Models](#models) | one section per model: corpus, recipe, results |
| [Upstream](#upstream) | tracking NVIDIA/Model-Optimizer |

## Install

```bash
git clone -b omk-main https://github.com/mann1x/omk-model-optimizer.git && cd omk-model-optimizer
bash omk/scripts/install-git-hooks.sh            # gitleaks pre-commit secret gate
bash omk_qad/conda_env/setup_conda_env.sh        # env omk-modelopt; ends with OMK_ENV_OK
```

Secrets: the pre-commit hook and the `omk-gitleaks` workflow scan every omk commit with
[`omk/gitleaks.toml`](../omk/gitleaks.toml) (default rules plus a block on literal `:-` defaults for secret-named
variables). Never
hardcode a token; fail loud with `${HF_TOKEN:?HF_TOKEN must be exported}`.

Also needed: a CUDA 12.x toolkit (≥ 12.8 for Blackwell) and a built llama.cpp checkout. Ternary files additionally
need the opencoti engine to run. Details: [`omk/docs/install.md`](../omk/docs/install.md).

## How it works

```bash
python -m omk_qad.multi --spec SPEC.yaml all     # plan -> cache -> preflight -> train -> export -> report
```

| phase | what |
|---|---|
| `plan` | per-student **layout**: llama.cpp's converter run on index probes gives every GGUF tensor's source rows and column order; `llama-quantize --dry-run <ftype>` gives each tensor's type. Classes can be fixed by hand (e.g. dense MLP at IQ2_XXS); the rest follows an ftype preset (`--rest IQ2_M`, `IQ3_M`, `Q3_K_M`, …). |
| `cache` | the teacher runs **once**: top-K logits for every training sequence, eval rows, and the imatrix. The teacher is never resident while students train. |
| `preflight` | each student alone for a few steps: peak memory, so students sharing a GPU are known to fit. |
| `train` | every quantized tensor is an fp16 latent **fake-quantized at its planned type** (ModelOpt encoders for IQ1/IQ2/Q8_0, libggml's own for the rest, PrismML's codec + Hadamard basis for ternary); Jaccard or KL distillation on the cached top-K; the MTP head distilled with the trunk detached. Pausable and resumable bit for bit. |
| `export` | latents → GGUF with the encoders training served; every quantized tensor is proven to be the rows training encoded, then re-read and checked against gguf-py's reference decoders. |
| `report` | one table per student and eval family. |

Module reference: [`omk_qad/README.md`](../omk_qad/README.md).

BF16 recovery skips `plan` and the GGUF types: `rollouts` (conversations → role-masked windows), `cache` (teacher
top-K, or only eval sets for SFT), `train --mode bf16`, `export --out-hf`. See
[`omk/docs/bf16-recovery.md`](../omk/docs/bf16-recovery.md).

## Recipes

A recipe is one `omk_qad.multi` spec: model, data mix, shared cache settings, and one entry per student (layout and
hyper-parameters).

| model | recipe | students |
|---|---|---|
| Qwen3.5-4B | [`omk/models/qwen3.5-4b/recipe_m6.yaml`](../omk/models/qwen3.5-4b/recipe_m6.yaml) | `iq2`: dense MLP IQ2_XXS + IQ2_M-rule rest, Jaccard, lr 1e-5 · `tern`: PQ2_0 MLP/attention/head, KL, lr 6e-5 · 2,400 steps × 8 × 1,024 tokens |
| Gemma-4-26B-A4B-it (agentic) | [`omk/models/gemma-4-26b-a4b/`](../omk/models/gemma-4-26b-a4b/README.md) (`sft11/`) | BF16 recovery, SFT (`--loss ce`): attention + dense MLP + router + routed experts, 2,108 steps × 2 × 16,384 tokens, lr 4.74e-6 |

## Datasets

[`omk/datasets/README.md`](../omk/datasets/README.md): every source (upstream, revision, how it was rendered, which
model used it) and the builders that stage and render them. Each model section pins its mix by sha256.

| source | kind | used in |
|---|---|---|
| AtomicChat `calib-corpora` (per-tokenizer builds) | calibration text: agentic, code, reasoning, multilingual, long context | Qwen3.5-4B |
| FineWeb-edu | general web text | Qwen3.5-4B |
| `corpus_v1` | the student model's own think / no-think rollouts | Qwen3.5-4B |
| NVIDIA Nemotron SFT (OpenCode, SWE, Competitive Programming) | agentic coding, SWE patches, competitive programming, SQL | Qwen3.5-4B |
| [`ManniX-ITA/gemma-4-26B-A4B-agentic-sft`](https://huggingface.co/datasets/ManniX-ITA/gemma-4-26B-A4B-agentic-sft) | on-policy SFT corpus: prompts / tools / tool outputs from 9 Nemotron sets + hermes, every trained turn regenerated by Gemma-4-26B-A4B-it itself; conversations + the exact 16k role windows trained | Gemma-4-26B-A4B agentic |
| [`ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b`](https://huggingface.co/datasets/ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b) | lambda Kimi-K2.5 agent tasks with the assistant turns regenerated by Gemma-4-31B-it | Gemma-4-26B-A4B agentic |

## Evaluations

[`omk/docs/evaluation.md`](../omk/docs/evaluation.md). Every model is measured the same way, students next to stock
llama.cpp quants of the same F16:

- **KLD** vs the F16 GGUF on wiki.test (100 × 512), with same top-p and PPL;
- **HumanEval-164 and HumanEval+**, greedy, raw completions through `llama-server`, every sample file checked;
- **stock baselines**: `quantize_gguf.py` tiers with the v5 calibration imatrix, same llama.cpp build;
- **engine controls** where a file needs a different engine (ternary).

## Models

### Qwen3.5-4B — [`omk/models/qwen3.5-4b/`](../omk/models/qwen3.5-4b/README.md)

Two students from one teacher on a ~20M-token code-heavy mix, one RTX PRO 6000 Blackwell (8.6 h / 6.0 h).

| file | size | HumanEval | HE+ | KLD vs F16 |
|---|---|---|---|---|
| F16 | 8.67 GB | 57.93 | 53.05 | — |
| Q4_K_M (stock) | 2.78 GB | 62.80 | 54.27 | 0.0271 |
| IQ2_S (stock) | 1.72 GB | 23.17 | 18.90 | 0.4019 |
| **omk_qad IQ2_XXS student** | **1.66 GB** | **38.41** | **31.71** | 0.3503 |
| IQ2_XXS (stock) | 1.61 GB | 10.37 | 9.76 | 0.6181 |
| **omk_qad ternary student (PQ2_0)** | **1.20 GB** | **17.68** | **15.85** | 0.9117 |

Full table (stock IQ1_M … Q4_K_M, off-grid re-encodes, packaging ladder), corpus catalogue and recipe in the model
section.

### Gemma-4-26B-A4B-it agentic — [`omk/models/gemma-4-26b-a4b/`](../omk/models/gemma-4-26b-a4b/README.md)

BF16 recovery SFT against reasoning / tool-call loops on long agentic tasks: an on-policy corpus (the model's own
regenerated turns, validator-checked, plus Gemma-4-31B regenerations of Kimi agent tasks), 4,217 × 16k windows, one
RTX PRO 6000 Blackwell, 25.5 h. Weights: [ManniX-ITA/gemma-4-26B-A4B-agentic](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic),
GGUF: [ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF).

| checkpoint (BF16 GGUF) | agentic gate pass (manic_miner, 10 runs) | loops / model-hour |
|---|---|---|
| base gemma-4-26B-A4B-it | 1 / 10 | 9.2 |
| **agentic (sft11 final)** | **5 / 10** | **2.0** |

Corpus build, every script as run, reproduce commands and GGUF build in the model section.

## Upstream

`main` is a fast-forward mirror of `NVIDIA/Model-Optimizer`; `omk-main` (default) is upstream plus omk's additive
tree, and never edits an upstream file. `bash omk/scripts/upstream_status.sh` reports upstream drift, the upstream
commits touching the ModelOpt modules `omk_qad` imports, and checks the additive rule. Procedure:
[`omk/docs/upstream.md`](../omk/docs/upstream.md).

## Repository layout

```
omk_qad/                 the package (layout, fakequant, train, export, multi, ternary, ...) + tests + conda_env/
omk/docs/                mode guides (qad, ternary, bf16-recovery, qad-ternary-together), install, evaluation, upstream
omk/datasets/            source catalogue + builders (stage_assets, build_mix, extract_stage2_prompts)
omk/models/<model>/      README (corpus, recipe, results), recipe spec, corpus/ manifests, evals/ (scripts, results)
omk/scripts/             upstream_status.sh, he_samples.py, install-git-hooks.sh
omk/gitleaks.toml        secret-scan config (hook + CI)
.github/README.md        this page
.github/workflows/omk-*  omk's CI (gitleaks); upstream's workflows are disabled on this repo
everything else          NVIDIA Model-Optimizer, unmodified
```

License: Apache-2.0, as upstream ([`LICENSE`](../LICENSE)).
