# SPDX-License-Identifier: Apache-2.0
"""Check that the running interpreter is a correct omk-modelopt env; print OMK_ENV_OK or exit 1.

    python omk_qad/conda_env/verify_env.py [--no-cuda]

Checks, in order (every failure is listed, nothing stops at the first):
  pins       every ``name==version`` in requirements-torch/requirements/requirements-build.txt is installed
  modelopt   ``import modelopt`` resolves to THIS checkout (editable install), not a PyPI wheel
  imports    omk_qad's modules and its third-party imports
  fast path  transformers' Qwen3.5 linear-attention kernels (causal-conv1d + fla) are all importable
  cuda       (unless --no-cuda) a GPU is visible; CUDA_HOME's nvcc has torch's CUDA MAJOR (>= 12.8 for sm_120);
             the GGML IQ packing extension builds and loads (raise_if_failed: the trainer otherwise WARNS and
             falls back to CPU encoding, 25x slower -- bug-822: /usr/local/cuda at 13.x while torch is cu128)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REQS = ("requirements-torch.txt", "requirements.txt", "requirements-build.txt")
IMPORTS = ("torch", "transformers", "safetensors", "numpy", "gguf", "pyarrow", "huggingface_hub", "accelerate",
           "yaml", "omegaconf", "pytest", "omk_qad.train", "omk_qad.multi", "omk_qad.layout", "omk_qad.fakequant",
           "omk_qad.gguf_export", "omk_qad.ptq", "modelopt.torch.quantization.ggml")


def _norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins() -> dict[str, str]:
    """{normalised name: version} of every ``==`` pin in the requirements files."""
    out = {}
    for f in REQS:
        for line in (HERE / f).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line and not line.startswith("-"):
                name, ver = line.split("==", 1)
                out[_norm(name)] = ver.strip()
    return out


def main() -> None:
    """Run every check; print OMK_ENV_OK or the failures and exit 1."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cuda", action="store_true", help="skip the GPU / nvcc / extension-build checks")
    a = ap.parse_args()
    bad: list[str] = []
    sys.path.insert(0, str(REPO))

    for name, want in pins().items():  # md.version resolves like import does (an env can hold stale dist-info)
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            bad.append(f"pin {name}=={want}: not installed")
            continue
        if have.split("+")[0] != want:
            bad.append(f"pin {name}=={want}: installed {have}")

    try:
        import modelopt

        if Path(modelopt.__file__).resolve().parents[1] != REPO:
            bad.append(f"modelopt resolves to {modelopt.__file__}, not this checkout {REPO} "
                       "(pip install --no-deps -e .)")
    except Exception as e:
        bad.append(f"import modelopt: {type(e).__name__}: {e}")

    for mod in IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: PERF203 -- one failure per module is the point
            bad.append(f"import {mod}: {type(e).__name__}: {str(e)[:200]}")

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

        if not q35.is_fast_path_available:
            bad.append("Qwen3.5 linear-attention fast path unavailable (causal-conv1d / fla kernels missing)")
    except Exception as e:
        bad.append(f"Qwen3.5 fast-path check: {type(e).__name__}: {e}")

    import torch

    print(f"python {sys.version.split()[0]} ({sys.executable})")
    print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
    if not a.no_cuda:
        if not torch.cuda.is_available():
            bad.append("no CUDA device visible (use --no-cuda for a CPU-only check)")
        cuda_home = os.environ.get("CUDA_HOME", "")
        nvcc = Path(cuda_home) / "bin" / "nvcc"
        if not cuda_home or not nvcc.exists():
            bad.append(f"CUDA_HOME={cuda_home!r} has no bin/nvcc; set it to the CUDA {torch.version.cuda} toolkit")
        else:
            out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=False).stdout
            m = re.search(r"release (\d+)\.(\d+)", out)
            tmaj = torch.version.cuda.split(".")[0]
            archs = [float(x) for x in re.findall(r"\d+\.\d+", os.environ.get("TORCH_CUDA_ARCH_LIST", ""))]
            if not m or m.group(1) != tmaj:  # bug-822: a different MAJOR (13.x vs torch cu12x) cannot build
                bad.append(f"nvcc {m.group(0) if m else '?'} at {nvcc}: major != torch CUDA {torch.version.cuda}")
            elif archs and max(archs) >= 12.0 and (int(m.group(1)), int(m.group(2))) < (12, 8):
                bad.append(f"nvcc {m.group(1)}.{m.group(2)} cannot target sm_120 "
                           f"(TORCH_CUDA_ARCH_LIST {archs}); need >= 12.8")
            else:
                print(f"nvcc {m.group(1)}.{m.group(2)} ({nvcc}); TORCH_CUDA_ARCH_LIST="
                      f"{os.environ.get('TORCH_CUDA_ARCH_LIST', '-')}")
        if not bad:
            try:
                from modelopt.torch.quantization.extensions import get_cuda_ext_ggml

                ext = get_cuda_ext_ggml(raise_if_failed=True)
                print("GGML IQ extension:", " ".join(sorted(n for n in dir(ext) if n.endswith("_pack"))))
            except Exception as e:
                bad.append(f"GGML IQ extension build: {type(e).__name__}: {str(e)[:300]}")

    if bad:
        print("OMK_ENV_FAIL")
        for b in bad:
            print("  -", b)
        raise SystemExit(1)
    print("OMK_ENV_OK")


if __name__ == "__main__":
    main()
