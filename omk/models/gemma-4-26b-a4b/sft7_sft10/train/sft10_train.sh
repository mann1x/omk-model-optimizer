#!/bin/bash
# bm-889 sft10 (owner 2026-11-09 23:3x "run an sft 2 pass on the corpus sft7+sft9, one single pass this time, not one
#   after another"; AskUserQuestion: de-duplicated union, keep cp_cf + if_rlvr, lr by sft7's rule, no source weights).
# From the UNTRAINED base on GPU0. Corpus sft7_build/v10/render_32k_v10m (V10_BUILD_OK, OVERLAP_OK, manifest
#   bm889-sft10-corpus-v10-union): corpus v7b (sft7's 119 rollouts with the 2026-11-08 suggestion fix, kimi31b 693,
#   cp_cf 109, if_rlvr 98) + sft9's 37 extras = 755 windows.
# 2 passes in ONE run: omk_sft10_code = omk_sft7_code + --sft-epochs (pass e = its own permutation seeded seed+e; pass 0 =
#   the single-pass order). Steps = 2 x windows. LR = 1e-2/steps if steps > 1000 else 1e-5 (the sft5/sft6/sft7 rule).
#   No --sft-weights. Flags otherwise sft7's. Snapshots every 250 (keep 2) so a snapshot falls near the end of pass 1.
# Disk: >= 200G free on /mnt/sdc at launch; pause below 100G, resume at >= 160G. Sentinels SFT10_START / SFT10_RESUME /
#   SFT10_OK / SFT10_FAIL / SFT10_DISK_PAUSE / SFT10_DISK_RESUME / SFT10_MEM_PAUSE.
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft10_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft10_v10_2p; LOG=train_sft10.log
R=sft7_build/v10/render_32k_v10m
EPOCHS=2
fail(){ echo "SFT10_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
grep -q "sft_epochs" omk_sft10_code/omk_qad/train.py || fail "omk_sft10_code has no --sft-epochs"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK"
[ -s $R/CORPUS_MANIFEST.json ] || fail "$R has no CORPUS_MANIFEST.json"
$PY - "$R" <<'PYC' || fail "manifest check"
import json, hashlib, sys, os
d = sys.argv[1]; m = json.load(open(d + '/CORPUS_MANIFEST.json'))
assert m['name'] == 'bm889-sft10-corpus-v10-union', m['name']
for f, v in m['files'].items():
    assert hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() == v['sha256'], f'{f} sha256 differs'
print('MANIFEST_OK', len(m['files']))
PYC
[ -s $CACHE/evalset_heldout.pt ] || fail "held-out eval set missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl | paste -sd,)
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" = 755 ] || fail "expected 755 train windows, found $N"
STEPS=$((EPOCHS * N))
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 200 ] || fail "only ${free}G free on /mnt/sdc (need 200 at launch)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT10_RESUME $(date -u +%FT%TZ) windows $N epochs $EPOCHS steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT10_START $(date -u +%FT%TZ) windows $N epochs $EPOCHS steps $STEPS lr $LR corpus $R (from base, no weights)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 32768 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --sft-epochs $EPOCHS --mtp freeze --eval-set heldout \
  --eval-every 250 --save-every 50 --snapshot-every 250 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft10-v10-2p >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft10_trainer.pid
dpause=""
while kill -0 $T 2>/dev/null; do
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT10_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ -d $S1 ] && [ "${free:-999}" -lt 100 ] && [ -z "$dpause" ] && [ -z "${paused:-}" ]; then
    echo "SFT10_DISK_PAUSE $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo pause > $S1/control; dpause=1
  elif [ -n "$dpause" ] && [ "${free:-0}" -ge 160 ]; then
    echo "SFT10_DISK_RESUME $(date -u +%FT%TZ) ${free}G free" | tee -a $LOG; echo run > $S1/control; dpause=""
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT10_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
