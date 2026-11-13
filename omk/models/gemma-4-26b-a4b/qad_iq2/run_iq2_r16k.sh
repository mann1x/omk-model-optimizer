#!/usr/bin/env bash
# bm-892 r16k driver: omk_qad.multi phases, assistant-only Gemma-4-26B-A4B agentic IQ2_XXS QAD on bs2 GPU0.
# Usage: g4_agentic_iq2_r16k_run.sh <phase>...   Sentinels: G4R16_PHASE_OK <ph> / G4R16_FAIL <ph> / G4R16_DONE.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; SPEC=$R/runs/g4_agentic_iq2_r16k.yaml
# bug-822: torch is cu128; the JIT GGML IQ extension must build with 12.8 or it falls back to the CPU encoder.
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/code/omk-qad-g4r16k
[ $# -gt 0 ] || { echo "G4R16_FAIL no phases given"; exit 2; }
cd $R || { echo "G4R16_FAIL cd $R"; exit 1; }
for ph in "$@"; do
  echo "[g4r16 $(date -u +%FT%TZ)] === $ph ==="
  /srv/ml/envs/envs/omk-modelopt/bin/python -W ignore::FutureWarning -m omk_qad.multi --spec $SPEC "$ph" \
    || { echo "[g4r16 $(date -u +%FT%TZ)] G4R16_FAIL $ph"; exit 1; }
  echo "[g4r16 $(date -u +%FT%TZ)] G4R16_PHASE_OK $ph"
done
echo "[g4r16 $(date -u +%FT%TZ)] G4R16_DONE $*"
