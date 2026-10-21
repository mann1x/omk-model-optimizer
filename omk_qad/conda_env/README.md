# omk-modelopt — the conda env omk_qad runs in

One env, built the same way on every host: `omk-modelopt`. It holds a CUDA 12.8 torch, the pinned omk_qad
and modelopt requirements, **this repo installed editable** (so `import modelopt` is this fork), and the
Qwen3.5/Qwen3.6 linear-attention kernels.

```bash
bash omk_qad/conda_env/setup_conda_env.sh              # create omk-modelopt and verify (ends with OMK_ENV_OK)
bash omk_qad/conda_env/setup_conda_env.sh --update     # re-apply the pins to an existing env
bash omk_qad/conda_env/setup_conda_env.sh --recreate   # delete and rebuild it
python omk_qad/conda_env/verify_env.py [--no-cuda]     # check any interpreter against this file set
```

Options: `--name` (default `omk-modelopt`), `--python` (3.11), `--cuda-home DIR`, `--arch 8.6|12.0|auto`,
`--no-cuda` (CPU-only box: skips causal-conv1d and the GPU checks).

## Files

| file | what |
|---|---|
| `requirements-torch.txt` | `torch==2.10.0` from the cu128 wheel index |
| `requirements.txt` | everything else, pinned: transformers / tokenizers / safetensors / gguf / pyarrow, fla, modelopt's own `[project].dependencies`, pytest |
| `requirements-build.txt` | `causal-conv1d`, compiled from source against the env's torch (`--no-build-isolation`) |
| `setup_conda_env.sh` | the build (below) |
| `verify_env.py` | pins, imports, fast path, CUDA toolkit, GGML IQ extension build → `OMK_ENV_OK` |

The pins are the solidpc env every omk_qad milestone was validated on. Change a pin here, never ad hoc in
one host's env, and re-run the CPU suite.

## What the setup does

1. Finds conda on `PATH`, `$CONDA_EXE`, or the install roots we use (`/root/anaconda3`, `/srv/ml/envs`,
   `/workspace/miniconda`). Refuses to touch an existing env unless `--update` or `--recreate` is passed.
2. Picks the CUDA toolkit used for source/JIT builds. It must be **CUDA 12.x** — torch is cu128 and a 13.x
   nvcc cannot compile against its headers — and **≥ 12.8 when building for sm_120** (Blackwell). The
   preference order is `--cuda-home`, `$CUDA_HOME`, `/usr/local/cuda-12.8`, then the newest `/usr/local/cuda-12.*`.
   `--arch auto` reads the GPUs' compute capability from `nvidia-smi`.
3. Fresh env from conda-forge python (no Anaconda channel ToS, no inherited packages), then torch,
   `requirements.txt`, `pip install --no-deps -e .`, and causal-conv1d for the chosen arch. It fails if a
   later step replaces torch.
4. Stores `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` as conda env vars, so `conda activate omk-modelopt` builds
   extensions with the right toolkit.
5. Runs `verify_env.py`.

## Running with it

Scripts call the interpreter by absolute path, which does **not** apply the conda env vars. Export them
yourself whenever modelopt may build a CUDA extension (the GGML IQ packers JIT-compile on first use):

```bash
ENV=$(conda info --base)/envs/omk-modelopt
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0   # the values setup printed for this host
$ENV/bin/python -m omk_qad.train ...
```

If the extension fails to build, modelopt only **warns** and the trainer silently falls back to CPU
encoding (~25× slower per step). Treat `GGML IQ CUDA packing extension is unavailable` in a log as a
failure. `verify_env.py` builds it with `raise_if_failed=True`.

The CPU test suite must hide the GPU. With a GPU visible, transformers takes the causal-conv1d CUDA path on
CPU tensors:

```bash
CUDA_VISIBLE_DEVICES= $ENV/bin/python -m pytest -o addopts="" omk_qad/tests/
```

## Also needed (not pip)

- **llama.cpp checkout + build** (`--llama-cpp`, default `/opt/llama.cpp`): `libggml-base.so`, `llama-quantize`,
  `llama-perplexity`, and its `conversion/` package (GGUF export and layout import it from the checkout).
- A **CUDA 12.x toolkit** as described above. The driver alone is not enough to build the extensions.

## Hosts

| host | conda | toolkit picked | arch |
|---|---|---|---|
| solidpc | `/root/anaconda3/bin/conda` | `/usr/local/cuda-12.6` (no 12.8 installed; `/usr/local/cuda` is 13.3) | 8.6 |
| bs2 | `/srv/ml/envs/condabin/conda` (= `/root/anaconda3`; on `PATH` for ssh commands since 2026-10-21) | `/usr/local/cuda-12.8` (`/usr/local/cuda` is 13.3) | 12.0 |
