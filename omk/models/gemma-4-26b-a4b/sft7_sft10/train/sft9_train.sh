#!/bin/bash
# bm-889 sft9 (council csl-2026-11-09-0855-6fdb accepted 2026-11-09; owner: no length caps, moderate p50 reduction).
# Copy of sft8_train.sh with only these differences:
#   - WARM START: --stage 2 --branch s$STEPS --init-from runs/g4_bf16_sft7/stage1/final.pt (weights only; fresh optimizer and
#     schedule); the copy's sha256 is checked against final.pt.sha256 before launch
#   - corpus sft7_build/v9/render_32k_v9m (manifest bm889-sft9-corpus-v9: 109 PASS convs <= 15 min, no per-turn cap, +kimi31b)
#   - --seq 32768; lr 5e-6 fixed (council: correction, not formation); STEPS from $1 (council 150 of 389 windows)
#   - eval/save/snapshot every 50, snapshot-keep 2; launch needs >= 200G (last.pt + 2 snapshots + final.pt ~ 196G peak)
#   - smoke folded in: the init eval must be <= 0.40 NLL (sft7 weights; base is ~0.45 on this eval split), else the
#     trainer is stopped before any checkpoint exists (SFT9_FAIL init)
# Sentinels SFT9_START / SFT9_INIT_OK / SFT9_RESUME / SFT9_OK / SFT9_FAIL / SFT9_DISK_PAUSE / SFT9_DISK_RESUME / SFT9_MEM_PAUSE.
set -uo pipefail
STEPS=${1:?usage: sft9_train.sh STEPS}
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft7b_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft9_v9; LOG=train_sft9.log; INIT=runs/g4_bf16_sft7/stage1/final.pt
R=sft7_build/v9/render_32k_v9m
fail(){ echo "SFT9_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage2/s$STEPS
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
assert m['name'] == 'bm889-sft9-corpus-v9', m['name']
for f, v in m['files'].items():
    assert hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() == v['sha256'], f'{f} sha256 differs'
print('MANIFEST_OK', m['name'], len(m['files']))
PYC
[ -s $CACHE/evalset_heldout.pt ] || fail "held-out eval set missing"
[ -s $INIT ] && [ -s $INIT.sha256 ] || fail "no $INIT (+ .sha256)"
[ -e "$S1/last.pt" ] || (cd $(dirname $INIT) && sha256sum -c --status final.pt.sha256) || fail "$INIT sha256 differs from final.pt.sha256"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl | paste -sd,)
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge "$STEPS" ] || fail "only $N train windows"
LR=5e-06
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ -e "$S1" ] || [ "$free" -ge 200 ] || fail "only ${free}G free on /mnt/sdc (need 200 at launch)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$S1/last.pt" ]; then echo "SFT9_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT9_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR init $INIT corpus $R" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --stage 2 --branch s$STEPS --init-from $INIT \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 32768 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-set heldout \
  --eval-every 50 --save-every 50 --snapshot-every 50 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft9-v9-sft7init-s$STEPS >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft9_trainer.pid
L0=$(wc -l < $LOG)
dpause=""
while kill -0 $T 2>/dev/null; do
  if [ -z "${initok:-}" ] && [ ! -e "$S1/last.pt.json" ]; then
    il=$(tail -n +$((L0 + 1)) $LOG | grep -m1 '^EVAL {"tag": "init"')
    if [ -n "$il" ]; then
      nll=$(echo "${il#EVAL }" | python3 -c 'import json,sys; print(json.load(sys.stdin)["nll"])')
      if awk -v x="$nll" 'BEGIN { exit !(x <= 0.40) }'; then echo "SFT9_INIT_OK init eval nll $nll (sft7 weights)" | tee -a $LOG; initok=1
      else kill $T; sleep 30; kill -0 $T 2>/dev/null && kill -9 $T; fail "init eval nll $nll > 0.40: not the sft7 weights? trainer stopped before any checkpoint"; fi
    fi
  fi
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT9_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ -d $S1 ] && [ "${free:-999}" -lt 55 ] && [ -z "$dpause" ] && [ -z "${paused:-}" ]; then
    echo "SFT9_DISK_PAUSE $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo pause > $S1/control; dpause=1
  elif [ -n "$dpause" ] && [ "${free:-0}" -ge 60 ]; then
    echo "SFT9_DISK_RESUME $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo run > $S1/control; dpause=""
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT9_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
