#!/bin/bash
# bm-889 sft11 (owner 2026-11-11 12:17 CET: "go ahead, bf16 like sft2, full epoch training, gate at 400 and then we decide
# if to stop or continue"). sft2's launcher (sft2_train2.sh) verbatim except: corpus = sft11_build/render (sft2 render_v3
# + kimi31b rendered the sft2 way, overlap gate 0, every window starts at <bos>), run/log/wandb names, W&B key read from
# /root/.bashrc (as sft10_train.sh), early hard links zero-padded (sft2's names never matched), pid file.
# Same code as sft2 (omk_sft2_code: bf16 latent storage with stochastic rounding, no fp32 master), same flags.
# Sentinels: SFT11_START / SFT11_RESUME / SFT11_OK / SFT11_FAIL. Usage: sft11_train.sh
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft2_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft11; R=sft11_build/render; LOG=train_sft11.log
fail(){ echo "SFT11_FAIL $*"; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK (sft2_overlap.py)"
[ -s $CACHE/evalset_heldout.pt ] || fail "held-out eval set missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl sft_build/sft_mix_v*_eval.roles.jsonl | grep -v fineweb | paste -sd,)
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge 200 ] || fail "only $N train windows"
STEPS=$(( N / 2 ))
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 300 ] || fail "only ${free}G free on /mnt/sdc"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT11_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT11_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 2 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-set heldout --eval-every 200 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft11 >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft11_trainer.pid
mkdir -p $RUN/early
while kill -0 $T 2>/dev/null; do
  for s in 0000100 0000200 0000400; do
    [ -e $S1/ckpt_step$s.pt ] && [ -e $S1/ckpt_step$s.json ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $S1/ckpt_step$s.json $RUN/early/
  done
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then  # 94 GiB of 97.9 reserved: checkpoint and stop before an OOM
    echo "SFT11_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT11_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
