#!/bin/bash
# bm-893 (owner 2026-11-14 11:34 CET: "let's use ml_k8 weights built from sft11 without shared alpha and re-run the sft11
# on it" + "use bs2 gpu1"). sft11_train.sh (bm-889 v11) verbatim except:
#   model  = the 98e ml_k8 no-α bf16 built from sft11 (/mnt/sdc/ml/v9a_sft11/mlk8-noa/..., map sha 84f2bd96)
#   GPU    = 1 (CUDA_VISIBLE_DEVICES and both nvidia-smi checks)
#   run/log/wandb/pid names
#   code   = omk_sft2_code_mlk8: omk_sft2_code (restored from archive/bm889_bs2_20261113, tree sha b7e6d0d0) with ONE
#            change: the cache's KD eval.pt is loaded only when it exists or the run is not SFT. eval.pt and
#            evalset_heldout.pt were not archived (*.pt excluded) and were deleted in the 11-13 cleanup. An SFT run never
#            uses eval.pt (run_eval -> sft_evaluate; the lossless gate is skipped for a 31B-GGUF cache).
#   no --eval-set heldout (its file is gone): the run evaluates on its own 20 held-out SFT files, like sft11's nll/ppl.
# Same corpus (sft11_build/render, 4,217 windows -> 2,108 steps, lr 4.74e-6), same flags otherwise.
# Sentinels: SFTM_START / SFTM_RESUME / SFTM_OK / SFTM_FAIL. Usage: sft11_mlk8noa_train.sh
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft2_code_mlk8 CUDA_VISIBLE_DEVICES=1
M=/mnt/sdc/ml/v9a_sft11/mlk8-noa/gemma-4-20B-v9-agentic-sft11-mlk8-noa; CACHE=cache_teacher_r4
RUN=runs/g4_bf16_sft11_mlk8noa; R=sft11_build/render; LOG=train_sft11_mlk8noa.log
fail(){ echo "SFTM_FAIL $*"; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -f "$M/config.json" ] && [ -f "$M/BUILD_OK" ] || fail "model $M missing or not BUILD_OK"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK (sft2_overlap.py)"
[ -s $CACHE/cache.json ] || fail "cache metadata missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl sft_build/sft_mix_v*_eval.roles.jsonl | grep -v fineweb | paste -sd,)
[ "$(echo $EVAL | tr , '\n' | wc -l)" -eq 20 ] || fail "expected sft11's 20 eval files"
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge 200 ] || fail "only $N train windows"
STEPS=$(( N / 2 ))
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 300 ] || fail "only ${free}G free on /mnt/sdc"
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU1 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFTM_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFTM_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 2 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-every 200 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm893-g4-98e-mlk8noa-sft11 >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft11_mlk8noa_trainer.pid
mkdir -p $RUN/early
while kill -0 $T 2>/dev/null; do
  for s in 0000100 0000200 0000400; do
    [ -e $S1/ckpt_step$s.pt ] && [ -e $S1/ckpt_step$s.json ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $S1/ckpt_step$s.json $RUN/early/
  done
  mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then  # 94 GiB of 97.9 reserved: checkpoint and stop before an OOM
    echo "SFTM_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFTM_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
