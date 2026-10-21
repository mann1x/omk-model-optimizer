#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# setup_conda_env.sh -- build the `omk-modelopt` conda env that omk_qad (and the modelopt fork it imports) runs in.
#
#   bash omk_qad/conda_env/setup_conda_env.sh [--name omk-modelopt] [--python 3.11] [--cuda-home DIR] [--arch 8.6|12.0|auto]
#                                       [--recreate | --update] [--no-cuda]
#
# Steps: fresh conda env (conda-forge python) -> torch cu128 (requirements-torch.txt) -> pinned requirements.txt
# -> this repo editable with --no-deps (modelopt) -> causal-conv1d built against the env's torch for --arch
# (requirements-build.txt) -> CUDA_HOME / TORCH_CUDA_ARCH_LIST stored as conda env vars -> verify_env.py.
# Ends with OMK_ENV_OK (or OMK_ENV_FAIL and exit 1). See README.md in this directory.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
NAME=omk-modelopt PY=3.11 CUDA_HOME_ARG="" ARCH=auto MODE=create NO_CUDA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --cuda-home) CUDA_HOME_ARG=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --recreate) MODE=recreate; shift ;;
    --update) MODE=update; shift ;;
    --no-cuda) NO_CUDA=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
say() { echo ">>> $(date -u +%H:%M:%S) $*"; }
die() { echo "OMK_ENV_FAIL: $*" >&2; exit 1; }
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1

# conda: PATH first, then the install roots used on our hosts (bs2's conda is not on PATH for non-login shells)
CONDA=${CONDA_EXE:-}
if [ -z "$CONDA" ] || [ ! -x "$CONDA" ]; then CONDA=$(command -v conda || true); fi
for c in /root/anaconda3/bin/conda /srv/ml/envs/bin/conda /workspace/miniconda/bin/conda "$HOME/miniconda3/bin/conda"; do
  [ -n "$CONDA" ] && break
  [ -x "$c" ] && CONDA=$c
done
[ -n "$CONDA" ] || die "no conda found (PATH, CONDA_EXE, /root/anaconda3, /srv/ml/envs, /workspace/miniconda)"
ROOT=$("$CONDA" info --base)
ENV=$ROOT/envs/$NAME
PYBIN=$ENV/bin/python
say "conda $("$CONDA" --version | cut -d' ' -f2) at $CONDA; env $ENV; mode $MODE"

# CUDA toolkit for source/JIT builds: torch is cu128, so the nvcc must be CUDA 12.x (bug-822: a 13.x nvcc fails on
# torch's headers) and >= 12.8 to target sm_120 (Blackwell). Preference: --cuda-home, $CUDA_HOME, 12.8, newest 12.x.
if [ "$NO_CUDA" = 0 ]; then
  if [ "$ARCH" = auto ]; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)
    [ -n "$ARCH" ] || die "--arch auto: nvidia-smi found no GPU (pass --arch, e.g. 8.6 or 12.0)"
  fi
  NEED=12.0; case ";$ARCH;" in *";12."*|*";1[3-9]."*) NEED=12.8 ;; esac
  rel() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'; }
  ok() { local r; r=$(rel "$1"); [ "${r%%.*}" = 12 ] && [ "$(printf '%s\n%s\n' "$NEED" "$r" | sort -V | head -1)" = "$NEED" ]; }
  CH=""
  # shellcheck disable=SC2012  # fixed /usr/local/cuda-12.N names
  for c in "$CUDA_HOME_ARG" "${CUDA_HOME:-}" /usr/local/cuda-12.8 $(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -rV); do
    if [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && ok "$c"; then CH=$c; break; fi
    if [ -n "$c" ] && [ "$c" = "$CUDA_HOME_ARG" ]; then die "--cuda-home $c: nvcc $(rel "$c") is not CUDA 12.x >= $NEED"; fi
  done
  [ -n "$CH" ] || die "no CUDA 12.x >= $NEED toolkit found for arch $ARCH (pass --cuda-home)"
  say "CUDA_HOME $CH (nvcc $(rel "$CH")); TORCH_CUDA_ARCH_LIST $ARCH"
fi

if [ -d "$ENV" ]; then
  case $MODE in
    create) die "$ENV exists; --update re-applies the pins in place, --recreate deletes and rebuilds it" ;;
    recreate) say "removing $ENV"; "$CONDA" env remove -y -q -n "$NAME" >/dev/null; [ ! -d "$ENV" ] || rm -rf "$ENV" ;;
    update) say "updating $ENV in place" ;;
  esac
fi
if [ ! -d "$ENV" ]; then
  say "conda create $NAME python=$PY (conda-forge)"
  "$CONDA" create -y -q -n "$NAME" --override-channels -c conda-forge "python=$PY" pip >/dev/null
fi
PIP=("$PYBIN" -m pip install -q)

say "torch (requirements-torch.txt)"
"${PIP[@]}" -r "$HERE/requirements-torch.txt"
TORCH=$("$PYBIN" -c 'import torch; print(torch.__version__)')
say "pinned requirements (requirements.txt)"
"${PIP[@]}" -r "$HERE/requirements.txt"
say "this repo, editable, --no-deps ($REPO)"
"${PIP[@]}" --no-deps -e "$REPO"
if [ "$NO_CUDA" = 0 ]; then
  say "causal-conv1d from source for $ARCH (requirements-build.txt)"
  CUDA_HOME=$CH PATH=$CH/bin:$PATH TORCH_CUDA_ARCH_LIST=$ARCH CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-16} \
    "${PIP[@]}" --no-build-isolation --no-deps -r "$HERE/requirements-build.txt"
  "$CONDA" env config vars set -n "$NAME" CUDA_HOME="$CH" TORCH_CUDA_ARCH_LIST="$ARCH" >/dev/null
fi
NOW=$("$PYBIN" -c 'import torch; print(torch.__version__)')
[ "$NOW" = "$TORCH" ] || die "a later step replaced torch $TORCH with $NOW"

say "verify"
if [ "$NO_CUDA" = 1 ]; then
  "$PYBIN" "$HERE/verify_env.py" --no-cuda
else
  CUDA_HOME=$CH PATH=$CH/bin:$PATH TORCH_CUDA_ARCH_LIST=$ARCH "$PYBIN" "$HERE/verify_env.py"
fi
