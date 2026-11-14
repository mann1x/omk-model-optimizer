#!/bin/bash
# bm-889 sft8 on corpus v8 (council csl-2026-11-08-2009-0c99; owner 2026-11-08 20:55 "v8 plus 33 at 16k", antirep in).
# Copy of sft7c_train.sh with only these differences:
#   - corpus sft7_build/v8/render_16k_v8m (manifest bm889-sft8-corpus-v8), code omk_sft7b_code (train.py = sft7's, checked)
#   - --seq 16384 (render seq), STEPS from $1 (council 300; one pass = 676 windows), lr by the same rule (<= 1000 -> 1e-5)
#   - run runs/g4_bf16_sft8_v8_s$STEPS, log train_sft8.log, eval/save/snapshot every 100, snapshot-keep 2
#   - disk: launch needs >= 210G on /mnt/sdc (last.pt + 2 snapshots + final.pt ~ 196G); guard pauses < 55G, resumes >= 60G
# No --sft-weights (owner 11:35 "drop weights"), no reasoning cap (council: selection, not caps).
# Sentinels SFT8_START / SFT8_RESUME / SFT8_OK / SFT8_FAIL / SFT8_DISK_PAUSE / SFT8_DISK_RESUME / SFT8_MEM_PAUSE.
set -uo pipefail
STEPS=${1:?usage: sft8_train.sh STEPS}
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft7b_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft8_v8_s$STEPS; LOG=train_sft8.log
R=sft7_build/v8/render_16k_v8m
fail(){ echo "SFT8_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
cmp -s omk_sft7_code/omk_qad/train.py omk_sft7b_code/omk_qad/train.py || fail "omk_sft7b_code train.py differs from sft7's"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK"
[ -s $R/CORPUS_MANIFEST.json ] || fail "$R has no CORPUS_MANIFEST.json"
$PY - "$R" <<'PYC' || fail "manifest check"
import json, hashlib, sys, os
d = sys.argv[1]; m = json.load(open(d + '/CORPUS_MANIFEST.json'))
assert m['name'] == 'bm889-sft8-corpus-v8', m['name']
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
[ -e "$RUN" ] || [ "$free" -ge 210 ] || fail "only ${free}G free on /mnt/sdc (need 210 at launch)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT8_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT8_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR weights none corpus $R" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-set heldout \
  --eval-every 100 --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft8-v8-s$STEPS >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft8_trainer.pid
dpause=""
while kill -0 $T 2>/dev/null; do
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT8_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ -d $S1 ] && [ "${free:-999}" -lt 55 ] && [ -z "$dpause" ] && [ -z "${paused:-}" ]; then
    echo "SFT8_DISK_PAUSE $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo pause > $S1/control; dpause=1
  elif [ -n "$dpause" ] && [ "${free:-0}" -ge 60 ]; then
    echo "SFT8_DISK_RESUME $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo run > $S1/control; dpause=""
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT8_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
