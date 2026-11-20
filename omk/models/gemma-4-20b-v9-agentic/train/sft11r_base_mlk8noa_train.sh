#!/bin/bash
# sft11r_base_mlk8noa_train.sh -- bm-898 follow-on (owner 2026-11-20 07:19 CET: "go with your defaults ... implement the
# router_full scope and telemetry, the load-balance term with the stop and second run gate, use bs2 gpu0 for the
# training"). sft11_base_mlk8noa_train.sh (the 11-18 run) with ONLY these changes:
#   scope   classes:attention,dense_mlp,router_full,routed_experts  -- the WHOLE router block trains: router.proj (as
#           before) + router.scale + router.per_expert_scale + the 5 FFN-block norm scales (fp32 params, --fp-lr)
#   --fp-lr 1e-4 (sft11 had it = the matrix LR 4.74e-6, which leaves O(1) gains where they are)
#   --router-gate 1.5,0.85,0.25: per eval, stop when a layer's top-8-of-98 load share > 1.5 x init, or its routing
#           entropy < 0.85 x init, or the share crosses 0.25 (init below) -> ROUTER_GATE.json, last.pt, SFTR_GATED
#   --lb-weight ${LB:-0}: 0 on the first run; the SECOND run (after a gate trip) sets LB=0.01 and a new RUN name
#   code    omk_sft11r_code (repo omk-main with omk_qad.router_telemetry), GPU0, run/log/wandb/pid names of its own
# Everything else is the sft11 recipe: same pruned no-α base from the 26B vendor model, same corpus (4,217 windows ->
# 2,108 steps, one epoch), lr 1e-2/steps = 4.74e-6, warmup 10, bs 2 x 16k, CE, mtp frozen, eval/200 on sft11's 20 eval
# files, snapshots/100 keep 2 (+ early 100/200/400 hard-linked). Telemetry per step: loss, lr, grad norm, update ratio,
# fp group update ratio + drift (router_scale / per_expert_scale / ffn_norm), lb term; per eval: ROUTER line (top-8
# share, entropy, top-1 flip vs init per layer) + the SFTB_EVAL family line.
# Sentinels: SFTR_START / SFTR_RESUME / SFTR_EVAL / SFTR_ROUTER / SFTR_MEM_PAUSE / SFTR_GATED / SFTR_OK / SFTR_FAIL.
# STORAGE (--bf16-latent): bf16 by default = what sft11 on this base ran with (its snapshot predates the fp16 default).
#   Measured 2026-11-20 on the real student, same rows, 3 steps: fp16 storage gives step-3 loss 0.841-0.842 (4 arms),
#   bf16 storage 0.821-0.823 (sft11, its snapshot, and this code) -- the scope and the telemetry do not move it. So the
#   router scope is the ONLY change against sft11 when STORAGE=bf16. The fp16 run paused at step 21 lives in the
#   untagged run dir and resumes only with STORAGE=fp16 RUN_TAG= (the trainer refuses a storage change on resume).
# Usage: RUN_TAG=bf16st sft11r_base_mlk8noa_train.sh   (first run, bf16 storage, lb 0)
#        LB=0.01 RUN_TAG=lb sft11r_base_mlk8noa_train.sh   (second run after a gate trip: its own run dir)
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft11r_code CUDA_VISIBLE_DEVICES=0
GPU=0
LB=${LB:-0}; RUN_TAG=${RUN_TAG:-}; STORAGE=${STORAGE:-bf16}
M=/mnt/sdc/ml/v9a_base/mlk8-noa/gemma-4-20B-v9-agentic-base-mlk8-noa; CACHE=cache_teacher_r4
RUN=runs/g4_bf16_sft11r_base_mlk8noa${RUN_TAG:+_$RUN_TAG}; R=sft11_build/render; LOG=train_sft11r_base_mlk8noa${RUN_TAG:+_$RUN_TAG}.log
fail(){ echo "SFTR_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e $S1/ROUTER_GATE.json ] && fail "$RUN was stopped by the router gate: start the second run with LB=0.01 RUN_TAG=lb"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -f "$M/config.json" ] && [ -f "$M/BUILD_OK" ] || fail "model $M missing or not BUILD_OK"
[ "$(cat "$(dirname "$M")/shared_alpha.txt" 2>/dev/null)" = none ] || fail "model is not the no-α build (shared_alpha.txt != none)"
[ -f $PYTHONPATH/omk_qad/router_telemetry.py ] || fail "code snapshot $PYTHONPATH lacks omk_qad/router_telemetry.py"
$PY -c "import omk_qad.router_telemetry, omk_qad.train" || fail "code snapshot does not import"
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
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU$GPU not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFTR_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR lb $LB" | tee -a $LOG
else echo "SFTR_START $(date -u +%FT%TZ) model $M windows $N steps $STEPS lr $LR fp-lr 1e-4 lb $LB storage $STORAGE gate 1.5,0.85,0.25 train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router_full,routed_experts --bs 2 --steps $STEPS --lr $LR --fp-lr 1e-4 --warmup 10 \
  --lb-weight $LB --router-gate 1.5,0.85,0.25 --router-telemetry on --bf16-latent $STORAGE \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-every 200 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-sft-20b-base-mlk8noa \
  --wandb-name bm898-g4-98e-base-mlk8noa-sft11r${RUN_TAG:+-$RUN_TAG} >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft11r_base_mlk8noa${RUN_TAG:+_$RUN_TAG}_trainer.pid
mkdir -p $RUN/early
seen=0; rseen=0
while kill -0 $T 2>/dev/null; do
  for s in 0000100 0000200 0000400; do
    [ -e $S1/ckpt_step$s.pt ] && [ -e $S1/ckpt_step$s.json ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $S1/ckpt_step$s.json $RUN/early/
  done
  mem=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then  # 94 GiB of 97.9 reserved: checkpoint and stop before an OOM
    echo "SFTR_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  n=$(grep -c '^EVAL ' $LOG)
  if [ "$n" -gt "$seen" ]; then
    seen=$n
    $PY - $LOG <<'PY' | tee -a $LOG
import json, sys
ev = [json.loads(l[5:]) for l in open(sys.argv[1], errors="ignore") if l.startswith("EVAL ")]
d = ev[-1]; f = d["families"]
fams = ["agentic", "opencode", "swe", "cp", "kimi31b", "hermes", "math", "science", "ifchat_v1", "ifchat_v2", "multilingual"]
print("SFTR_EVAL", d["tag"], "ppl %.3f" % d["ppl"], " ".join("%s %.3f" % (x, f[x + ".eval"]["nll"]) for x in fams if x + ".eval" in f))
PY
  fi
  rn=$(grep -c '^ROUTER ' $LOG)
  if [ "$rn" -gt "$rseen" ]; then
    rseen=$rn
    $PY - $LOG <<'PY' | tee -a $LOG
import json, sys
rs = [json.loads(l[7:]) for l in open(sys.argv[1], errors="ignore") if l.startswith("ROUTER ")]
d = rs[-1]
worst = max(d["layers"].items(), key=lambda kv: kv[1]["top_share"])
print("SFTR_ROUTER n=%d top%d share max %.3f (L%s) mean %.3f entropy min %.3f mean %.3f flip mean %.3f max %.3f" % (
    len(rs), d["top_n"], d["top_share_max"], worst[0], d["top_share_mean"], d["entropy_min"], d["entropy_mean"],
    d.get("flip_mean", 0.0), d.get("flip_max", 0.0)))
PY
  fi
  sleep 30
done
wait $T
if [ -e $S1/ROUTER_GATE.json ]; then
  echo "SFTR_GATED $(date -u +%FT%TZ) $(grep -o '"step": [0-9]*' $S1/ROUTER_GATE.json | head -1) reasons: $(grep ROUTER_GATE_STOP $LOG | tail -1 | cut -c1-400)" | tee -a $LOG
  exit 2
fi
if grep -q '"status": "\(paused\|drained\)"' $S1/state.json 2>/dev/null; then
  echo "SFTR_PAUSED $(date -u +%FT%TZ) $(grep -o '"step": [0-9]*' $S1/state.json) -- relaunch the same command to resume" | tee -a $LOG
  exit 3
fi
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFTR_OK $(date -u +%FT%TZ) steps $STEPS lb $LB" | tee -a $LOG
