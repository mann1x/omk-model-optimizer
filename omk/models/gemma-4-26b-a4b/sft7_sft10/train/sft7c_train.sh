#!/bin/bash
# bm-889 sft7c 200-step arm (owner 2026-11-08 11:42 CET "run the probe, then launch the 200-step arm"; council
# csl-2026-11-08-1024-f89f: capped v7b render, all source weights 1.0, 200 steps, then the length battery).
# Copy of sft7_train.sh with only these differences:
#   - corpus sft7_build/render_32k_v7cm (v7b + --score-reasoning-cap 1000), code omk_sft7c_code (train.py = sft7's)
#   - NO --sft-weights (owner 11:35 "drop weights": every source 1.0)
#   - steps 200 (not 1 pass); lr by the same rule (<= 1000 steps -> 1e-5); eval every 100
#   - run runs/g4_bf16_sft7c_s200, log train_sft7c.log, no early pins (snapshots 100 + 200 kept)
#   - disk: launch needs >= 200G on /mnt/sdc (peak ~4 x 49G); the guard pauses only below 55G (a save needs 49G) and
#     resumes at >= 60G, so it cannot sit paused for hours as sft7's 100G/160G guard did (bug-908)
# Sentinels: SFT7C_START / SFT7C_RESUME / SFT7C_OK / SFT7C_FAIL / SFT7C_DISK_PAUSE / SFT7C_DISK_RESUME / SFT7C_MEM_PAUSE.
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft7c_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft7c_s200; LOG=train_sft7c.log
R=sft7_build/render_32k_v7cm
STEPS=200
fail(){ echo "SFT7C_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
cmp -s omk_sft7_code/omk_qad/train.py omk_sft7c_code/omk_qad/train.py || fail "omk_sft7c_code train.py differs from sft7's"
grep -q "REASONING_CAP" omk_sft7c_code/omk_qad/rollouts.py || fail "omk_sft7c_code has no reasoning cap"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK"
[ -s $R/CORPUS_MANIFEST.json ] || fail "$R has no CORPUS_MANIFEST.json"
$PY - "$R" <<'PYC' || fail "manifest check"
import json, hashlib, sys, os
d = sys.argv[1]; m = json.load(open(d + '/CORPUS_MANIFEST.json'))
assert m['name'] == 'bm889-sft7-corpus-v7c', m['name']
for f, v in m['files'].items():
    assert hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() == v['sha256'], f'{f} sha256 differs'
print('MANIFEST_OK', m['name'], len(m['files']))
PYC
[ -s $CACHE/evalset_heldout.pt ] || fail "held-out eval set missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl | paste -sd,)
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge "$STEPS" ] || fail "only $N train windows"
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ -e "$RUN" ] || [ "$free" -ge 200 ] || fail "only ${free}G free on /mnt/sdc (need 200 at launch)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT7C_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT7C_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR weights none (1.0) corpus $R" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 32768 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-set heldout \
  --eval-every 100 --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft7c-s200 >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft7c_trainer.pid
dpause=""
while kill -0 $T 2>/dev/null; do
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT7C_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ -d $S1 ] && [ "${free:-999}" -lt 55 ] && [ -z "$dpause" ] && [ -z "${paused:-}" ]; then
    echo "SFT7C_DISK_PAUSE $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo pause > $S1/control; dpause=1
  elif [ -n "$dpause" ] && [ "${free:-0}" -ge 60 ]; then
    echo "SFT7C_DISK_RESUME $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo run > $S1/control; dpause=""
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT7C_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
