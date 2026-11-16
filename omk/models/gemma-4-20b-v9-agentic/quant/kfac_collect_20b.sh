#!/bin/bash
# bm-893: K-FAC factors of the 20B v9-agentic (sft12) — the same recipe as the 26B g10 run
# (omk_qad.kfac, 512 windows x 2048 tokens, seed 1, diagonal down-G, dense MLP, absolute |g| row cut 10).
# Token windows: the 26B QAD cache (same Gemma tokenizer; only x/role are used -- labels are sampled from the
# 20B's own distribution, the true Fisher). Factors ~59 GB (98 experts). Sentinels KFAC20_START/DONE/FAIL.
set -uo pipefail
GPU=${1:?gpu index}
M=/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12
R=/mnt/sdc/omk-model-optimizer/runs/g4_20b_kfac; mkdir -p $R; LOG=$R/collect_g10.log
OUT=$R/factors_w512_s2048_g10
PY=/srv/ml/envs/envs/omk-modelopt/bin/python; OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
export PYTHONPATH=$OMK CUDA_VISIBLE_DEVICES=$GPU
fail(){ echo "KFAC20_FAIL $(date -u +%FT%TZ) $*" | tee -a $LOG; exit 1; }
[ -s $M/config.json ] || fail "no model at $M"
[ -e $OUT ] && fail "$OUT exists (never overwritten)"
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU$GPU not free ($used MiB used)"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 130 ] || fail "only ${free}G free on /mnt/sdc (need 130G: factors ~59G + headroom)"
echo "KFAC20_START $(date -u +%FT%TZ) gpu $GPU model $M free ${free}G out $OUT" | tee -a $LOG
$PY -m omk_qad.kfac --model $M \
  --cache /mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k/cache --out $OUT --device cuda:0 \
  --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10 \
  >> $LOG 2>&1 < /dev/null || fail "kfac: $(tail -5 $LOG | tr '\n' ' ' | cut -c1-400)"
grep -q "KFAC_DONE" $LOG || fail "no KFAC_DONE"
echo "KFAC20_DONE $(date -u +%FT%TZ) $OUT $(du -sh $OUT | cut -f1)" | tee -a $LOG
