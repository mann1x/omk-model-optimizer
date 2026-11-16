#!/bin/bash
# bm-895 recollection against the attention-sink inflation (owner 2026-11-15 21:39: "if the residual rule holds,
# recollect all 30 layers"). Same recipe as kfac_collect.sh (512 windows x 2048 tokens, 3 layers/pass, down G diagonal,
# dense) plus ONE of:
#   K = <number> → omk_qad.kfac --drop-resid K (residual-norm exclusion; 22:10: did NOT identify the sinks, drops nothing)
#   K = rad      → omk_qad.kfac --rademacher (random-sign per-loss Fisher; 22:25: REFUTED, no effect)
#   K = g<G>     → omk_qad.kfac --drop-grow G (ABSOLUTE per-row |g| cut; owner 23:26 "ok A" = g10 on all 30 layers)
# Output goes to a NEW dir (factors_w512_s2048_<mode>); the first factor set is never touched.
# Disk: a second 77 GB set next to the running QAD's 60 GB snapshot transient does not fit in 140 GB on /mnt/sdc, so
# with --after-pid <trainer pid> the launcher waits for the QAD to exit and then needs >= 90 GB; without it >= 150 GB.
# Usage: kfac_collect_r.sh <gpu> <K> [--after-pid PID]   Sentinels KFACRUN_START / KFACRUN_OK / KFACRUN_FAIL.
set -uo pipefail
GPU=${1:?gpu index}; K=${2:?mode: <number> (--drop-resid), rad (--rademacher) or g<G> (--drop-grow G)}; shift 2
case "$K" in
  rad) MODE="--rademacher"; TAG=rad ;;
  g*)  MODE="--drop-grow ${K#g}"; TAG=$K ;;
  *)   MODE="--drop-resid $K"; TAG=r$K ;;
esac
AFTER=0; [ "${1:-}" = "--after-pid" ] && AFTER=${2:?pid}
R=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac; mkdir -p $R; LOG=$R/collect_$TAG.log
OUT=$R/factors_w512_s2048_$TAG
PY=/srv/ml/envs/envs/omk-modelopt/bin/python; OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
export PYTHONPATH=$OMK CUDA_VISIBLE_DEVICES=$GPU
fail(){ echo "KFACRUN_FAIL $*" | tee -a $LOG; exit 1; }
if [ "$AFTER" != 0 ]; then
  echo "KFACRUN_WAIT $(date -u +%FT%TZ) for pid $AFTER (the QAD trainer) to exit" | tee -a $LOG
  while kill -0 "$AFTER" 2>/dev/null; do sleep 30; done
  NEED=90
else
  NEED=150
fi
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU$GPU not free ($used MiB used)"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge "$NEED" ] || fail "only ${free}G free on /mnt/sdc (need ${NEED}G: factors ~77G + headroom)"
echo "KFACRUN_START $(date -u +%FT%TZ) gpu $GPU mode [$MODE] free ${free}G out $OUT" | tee -a $LOG
$PY -m omk_qad.kfac --model /mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic \
  --cache /mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k/cache --out $OUT --device cuda:0 \
  --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense $MODE \
  >> $LOG 2>&1 < /dev/null || fail "kfac: $(tail -5 $LOG | tr '\n' ' ' | cut -c1-400)"
grep -q "KFAC_DONE" $LOG || fail "no KFAC_DONE"
echo "KFACRUN_OK $(date -u +%FT%TZ) $(du -sh $OUT | cut -f1)" | tee -a $LOG
