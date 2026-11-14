#!/bin/bash
# bm-889 sft7 (owner 2026-11-07 08:56 CET "stop the sft6 training and restart it with whole conversation with full
# rethink, per source loss weights, 1 pass and adjusted lr"; 09:3x weights 70/10/10/10). From the untrained base on GPU0.
# Corpus: sft7_build/render_32k_v2m (CORPUS_MANIFEST.json, OVERLAP_OK): whole conversations, every turn's reasoning,
# [SYSTEM] lines dropped. Code: omk_sft7_code (--sft-weights). Recipe otherwise sft6's (sft5's trainer flags).
# Steps = train windows (1 pass). LR = 1e-2/steps if steps > 1000 else 1e-5 (the sft5/sft6 rule).
# Weights (gradient only; logged loss + guard unweighted): target loss shares rollouts 70 %, kimi31b / cp_cf / if_rlvr
# 10 % each, from untrained per-token NLL (sft6 init eval) x scored tokens -> 0.813 / 1.81 / 2.01 / 2.99.
# Disk: the sft6 run is being moved to solidpc (sft6_run_archive.sh); launch needs >= 150G on /mnt/sdc, the watch
# loop pauses the trainer below 100G free and resumes it at >= 160G. Peak resident: early pins 200 + 400, 2 snapshots,
# last, 1 temp = 6 x 49 GB.
# Sentinels: SFT7_START / SFT7_RESUME / SFT7_OK / SFT7_FAIL / SFT7_DISK_PAUSE / SFT7_DISK_RESUME / SFT7_MEM_PAUSE.
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft7_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft7; LOG=train_sft7.log
R=sft7_build/render_32k_v2m
W="rollouts_s0.train=0.813,rollouts_s1.train=0.813,rollouts_s2.train=0.813,rollouts_s3.train=0.813,kimi31b.train=1.81,cp_cf.train=2.01,if_rlvr.train=2.99"
fail(){ echo "SFT7_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
grep -q "sft_weights" omk_sft7_code/omk_qad/train.py || fail "omk_sft7_code has no --sft-weights"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK"
[ -s $R/CORPUS_MANIFEST.json ] || fail "$R has no CORPUS_MANIFEST.json"
$PY - "$R" <<'PYC' || fail "manifest check"
import json, hashlib, sys, os
d = sys.argv[1]; m = json.load(open(d + '/CORPUS_MANIFEST.json'))
for f, v in m['files'].items():
    assert hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() == v['sha256'], f'{f} sha256 differs'
print('MANIFEST_OK', len(m['files']))
PYC
[ -s $CACHE/evalset_heldout.pt ] || fail "held-out eval set missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl | paste -sd,)
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge 200 ] || fail "only $N train windows"
STEPS=$N
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 150 ] || fail "only ${free}G free on /mnt/sdc (need 150 at launch; the sft6 archive frees the rest)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT7_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT7_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR weights $W" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 32768 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --sft-weights "$W" --mtp freeze --eval-set heldout \
  --eval-every 200 --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft7 >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft7_trainer.pid
mkdir -p $RUN/early
dpause=""
while kill -0 $T 2>/dev/null; do
  for s in 0000200 0000400; do [ -e $S1/ckpt_step$s.pt ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $RUN/early/; done
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT7_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ -d $S1 ] && [ "${free:-999}" -lt 100 ] && [ -z "$dpause" ] && [ -z "${paused:-}" ]; then
    echo "SFT7_DISK_PAUSE $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo pause > $S1/control; dpause=1
  elif [ -n "$dpause" ] && [ "${free:-0}" -ge 160 ]; then
    echo "SFT7_DISK_RESUME $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo run > $S1/control; dpause=""
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT7_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
