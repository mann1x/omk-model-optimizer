#!/bin/bash
# bm-898 (owner 2026-11-18 12:56 CET: "I'm not really satisfied with v9-agentic-sft12 ... Instead of doing sft11 and then
# sft12 on the prune, we test sft11 on the pruned 20b mlk8 without shared alpha starting from 26b base not the sft11").
# sft11_mlk8noa_train.sh (bm-893, itself sft11_train.sh verbatim) with ONLY these changes:
#   model  = the 98e ml_k8 no-α bf16 pruned from the 26B VENDOR BASE (/mnt/sdc/ml/v9a_base/mlk8-noa/..., map sha 84f2bd96,
#            built by build_v9_agentic_sft11_quick.sh with SRC=/mnt/sdc/ml/models/gemma-4-26B-A4B-it ALPHA=none)
#   run/log/wandb/pid names; W&B project of its own (omk-sft-20b-base-mlk8noa), per the one-project-per-workstream rule
#   SFTB_EVAL: one compact line per EVAL (ppl + per-family nll) so the eval curve can be read from the log/waiter
# Everything else is the sft11 recipe: corpus sft11_build/render (4,217 windows -> 2,108 steps = one epoch), lr 1e-2/steps
# = 4.74e-6, warmup 10, bs 2 x 16k, CE, scope attention+dense_mlp+router+routed_experts, mtp frozen, eval/200 on sft11's 20
# eval files, snapshots/100 keep 2 (+ early 100/200/400 hard-linked), GPU1. No stop rule: the init is an un-tuned base,
# so "regressed vs init" has no meaning here; the owner reads the SFTB_EVAL lines against the reference table in
# docs/plans/plan_v9_agentic_base_mlk8noa_sft11.md.
# Sentinels: SFTB_START / SFTB_RESUME / SFTB_EVAL / SFTB_MEM_PAUSE / SFTB_OK / SFTB_FAIL. Usage: sft11_base_mlk8noa_train.sh
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft2_code_mlk8 CUDA_VISIBLE_DEVICES=1
M=/mnt/sdc/ml/v9a_base/mlk8-noa/gemma-4-20B-v9-agentic-base-mlk8-noa; CACHE=cache_teacher_r4
RUN=runs/g4_bf16_sft11_base_mlk8noa; R=sft11_build/render; LOG=train_sft11_base_mlk8noa.log
fail(){ echo "SFTB_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -f "$M/config.json" ] && [ -f "$M/BUILD_OK" ] || fail "model $M missing or not BUILD_OK"
[ "$(cat "$(dirname "$M")/shared_alpha.txt" 2>/dev/null)" = none ] || fail "model is not the no-α build (shared_alpha.txt != none)"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK (sft2_overlap.py)"
[ -s $CACHE/cache.json ] || fail "cache metadata missing"
DATA=$(ls $R/*.train.roles.jsonl | paste -sd,)
EVAL=$(ls $R/*.eval.roles.jsonl sft_build/sft_mix_v*_eval.roles.jsonl | grep -v fineweb | paste -sd,)
[ "$(echo $DATA | tr , '\n' | wc -l)" -eq 11 ] || fail "expected sft11's 11 train files"
[ "$(echo $EVAL | tr , '\n' | wc -l)" -eq 20 ] || fail "expected sft11's 20 eval files"
for f in ${DATA//,/ } ${EVAL//,/ }; do grep -q '"complete": true' ${f%.jsonl}.stats.json || fail "incomplete render $f"; done
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -eq 4217 ] || fail "expected sft11's 4217 train windows, found $N"
STEPS=$(( N / 2 ))
LR=$(awk -v s=$STEPS 'BEGIN { printf "%.3g", (s > 1000 ? 1e-2 / s : 1e-5) }')
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 300 ] || fail "only ${free}G free on /mnt/sdc"
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU1 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFTB_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFTB_START $(date -u +%FT%TZ) model $M windows $N steps $STEPS lr $LR train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 2 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-every 200 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-sft-20b-base-mlk8noa \
  --wandb-name bm898-g4-98e-base-mlk8noa-sft11 >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft11_base_mlk8noa_trainer.pid
mkdir -p $RUN/early
seen=0
while kill -0 $T 2>/dev/null; do
  for s in 0000100 0000200 0000400; do
    [ -e $S1/ckpt_step$s.pt ] && [ -e $S1/ckpt_step$s.json ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $S1/ckpt_step$s.json $RUN/early/
  done
  mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then  # 94 GiB of 97.9 reserved: checkpoint and stop before an OOM
    echo "SFTB_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  n=$(grep -c '^EVAL ' $LOG)
  if [ "$n" -gt "$seen" ]; then
    seen=$n
    $PY - $LOG <<'PY' | tee -a $LOG
import json, sys
ev = [json.loads(l[5:]) for l in open(sys.argv[1], errors="ignore") if l.startswith("EVAL ")]
d = ev[-1]; f = d["families"]
fams = ["agentic", "opencode", "swe", "cp", "kimi31b", "hermes", "math", "science", "ifchat_v1", "ifchat_v2", "multilingual"]
print("SFTB_EVAL", d["tag"], "ppl %.3f" % d["ppl"], " ".join("%s %.3f" % (x, f[x + ".eval"]["nll"]) for x in fams if x + ".eval" in f))
PY
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFTB_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
