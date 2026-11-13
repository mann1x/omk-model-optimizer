#!/usr/bin/env bash
# bm-892 driver: omk_qad.multi phases for the Gemma-4-26B-A4B agentic IQ2_XXS QAD on bs2 GPU0.
# Usage: run_iq2.sh <phase>...   (first launch: plan cache preflight; then train export report once steps are fixed)
# Sentinels: G4IQ2_PHASE_OK <ph> / G4IQ2_FAIL <ph> / G4IQ2_DONE.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; SPEC=$R/runs/g4_agentic_iq2.yaml
# bug-822: /usr/local/cuda is 13.x but torch is cu128; the JIT GGML IQ extension must build with 12.8 or it silently
# falls back to the CPU encoder (300-415 s/step).
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/code/omk-qad-g4
[ $# -gt 0 ] || { echo "G4IQ2_FAIL no phases given"; exit 2; }
cd $R || { echo "G4IQ2_FAIL cd $R"; exit 1; }
for ph in "$@"; do
  echo "[g4iq2 $(date -u +%FT%TZ)] === $ph ==="
  /srv/ml/envs/envs/omk-modelopt/bin/python -W ignore::FutureWarning -m omk_qad.multi --spec $SPEC "$ph" \
    || { echo "[g4iq2 $(date -u +%FT%TZ)] G4IQ2_FAIL $ph"; exit 1; }
  echo "[g4iq2 $(date -u +%FT%TZ)] G4IQ2_PHASE_OK $ph"
done
echo "[g4iq2 $(date -u +%FT%TZ)] G4IQ2_DONE $*"
