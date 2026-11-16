#!/usr/bin/env bash
# kfac_ladder_lane.sh — RUNS ON bs2. One lane of the K-FAC ladder build (owner 2026-11-16 20:25 CET: "build all the 20b
# tiers with k-fac. and we republish and expand also the 26b agentic quants with kfac. leave out only for both the
# iq2_xxs, we will use the qad for both"). Every tier goes through kfac_ladder_twins.sh <model> <TIER> all: llama-quantize
# PTQ with the AC imatrix + attention rule, K-FAC re-rounding of every MLP tensor at the same types, 44-row KL twin score
# (the per-tier K-FAC vs llama.cpp table), ptq.gguf removed under the owner's .rm_ptq_granted marker. Runs beside the
# kfac_twins_driver.sh lane (disjoint tiers): CPU 93 % idle, GPU0 86 GB free, 3 rounders fit.
#   kfac_ladder_lane.sh <lane-name> <26b|20b> TIER...
# Disk: waits (never dies) while /mnt/sdc < $MIN_FREE_G, logging LN_DISK_WAIT every 10 min; the archiver on solidpc
# (kfac_ladder_archive.sh) frees space only with the owner's .rm_kfac_archived_granted marker.
# A tier whose kfac.gguf + kfac.eval.json exist is skipped; one whose dir holds a ptq.gguf or kfac.log without a score is
# another lane's work in progress, skipped too. A failing tier is logged (LN_TIER_FAIL) and the lane moves on.
# Sentinels LN_START / LN_STEP / LN_DISK_WAIT / LN_TIER_FAIL / LN_DONE in $LOG.
set -u
LANE=${1:?lane}; WHICH=${2:?26b|20b}; shift 2
RUNS=/mnt/sdc/omk-model-optimizer/runs
TW=$RUNS/kfac_ladder_twins.sh
case $WHICH in 26b) K=$RUNS/g4_agentic_kfac;; 20b) K=$RUNS/g4_20b_kfac;; *) echo "which=$WHICH"; exit 1;; esac
LOG=$RUNS/kfac_ladder_lane_$LANE.log
MIN_FREE_G=${MIN_FREE_G:-70}
STAGE=${STAGE:-all}   # all | build (ptq then kfac, no GPU0 scoring: used while the 20B QAD cache holds 85 GB of GPU0)
say() { echo "[ln $LANE $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
disk() { df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9; }
say "LN_START $WHICH stage $STAGE min_free ${MIN_FREE_G}G tiers: $*"
fails=""
for T in "$@"; do
  D=$K/ladder/$T
  if [ -s $D/kfac.gguf ] && [ -s $D/kfac.eval.json ]; then say "$T: done already, skip"; continue; fi
  # in progress = a twins process for this exact tier is RUNNING (bracket pattern never matches this grep itself);
  # stale ptq.log / ptq.partial.gguf from a killed run are overwritten by llama-quantize, not a reason to skip
  if ps -eo args | grep -q "[k]fac_ladder_twins.sh $WHICH $T "; then say "$T: another lane is on it, skip"; continue; fi
  n=0
  while [ "$(disk)" -lt $MIN_FREE_G ]; do
    [ $((n % 20)) = 0 ] && say "LN_DISK_WAIT /mnt/sdc $(disk)G < ${MIN_FREE_G}G before $WHICH $T (archiver + owner's rm grant free it)"
    n=$((n + 1)); sleep 30
  done
  say "LN_STEP $WHICH $T ($(disk)G free)"
  if [ "$STAGE" = build ]; then ok=0; bash $TW $WHICH $T ptq && bash $TW $WHICH $T kfac && ok=1; else ok=0; bash $TW $WHICH $T all && ok=1; fi
  if [ $ok = 1 ]; then
    say "$T: $(grep -E "LT_TWIN $T |LT_KFAC" $K/ladder/twins.log | tail -1 | cut -c1-200)"
  else
    say "LN_TIER_FAIL $WHICH $T: $(grep -E "LT_FAIL" $K/ladder/twins.log | tail -1 | cut -c1-200)"; fails="$fails $T"
  fi
done
say "LN_DONE $WHICH failed:[${fails# }]"
