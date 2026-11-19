# Install

`omk_qad` runs in one conda env, **`omk-modelopt`**, with this repo installed editable (so `import modelopt` is this
fork). The full reference is [`omk_qad/conda_env/README.md`](../../omk_qad/conda_env/README.md); the short form:

```bash
git clone -b omk-main https://github.com/mann1x/omk-model-optimizer.git && cd omk-model-optimizer
bash omk/scripts/install-git-hooks.sh                # gitleaks pre-commit secret gate (needs gitleaks on PATH)
bash omk_qad/conda_env/setup_conda_env.sh            # creates omk-modelopt, ends with OMK_ENV_OK
# options: --arch 8.6|12.0|auto  --cuda-home DIR  --update  --recreate  --no-cuda
```

Requirements outside pip:

| what | why |
|---|---|
| NVIDIA driver + **CUDA 12.x toolkit** (≥ 12.8 for Blackwell / sm_120) | torch is cu128; the GGML IQ packers and causal-conv1d compile against it. A 13.x nvcc cannot. |
| **llama.cpp checkout, built** (`--llama-cpp`, default `/opt/llama.cpp`) | `libggml-base.so` (encoders ModelOpt lacks), `llama-quantize` (layout dry runs), `llama-perplexity`, and its `conversion/` package (GGUF export) |
| opencoti engine + its `perplexity` build | only for ternary (PQ2_0 / PTQ1_0) files: stock llama.cpp cannot load them |

Running: scripts call the env's interpreter by absolute path, which skips the conda env vars, so export them:

```bash
ENV=$(conda info --base)/envs/omk-modelopt
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0    # what setup printed for this host
export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ENV/bin/python -m omk_qad.multi --spec SPEC.yaml all
```

`GGML IQ CUDA packing extension is unavailable` in a log is a failure (the trainer falls back to CPU encoding,
~25× slower). Tests: see **[testing.md](testing.md)** — validate a change with its impacted tests
(`omk_qad/tests/omk_test.sh impact` / `run NAME --impact`), run suites or single tests, pause / resume named runs;
the full 2.5 h CPU suite (`omk_test.sh run full`) is for the end of a stint, an upstream sync, or on demand.
`-o addopts=""` is mandatory for a bare pytest call (the runner adds it). Every failure is logged **as it happens**
(`OMK_TEST_FAILED <nodeid>` + traceback tail), so a killed run still names what failed.
