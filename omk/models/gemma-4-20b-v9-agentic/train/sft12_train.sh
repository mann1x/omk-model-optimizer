#!/bin/bash
# bm-893 sft12-98e (owner 2026-11-14 20:27 CET: "go, build the sft12 corpus, stop the current sft"; plan: "less corpus,
# more mild, different LR, no long traces"). sft11_mlk8noa_train.sh with these changes, each from the measured diagnosis
# (bm-893 note 2026-11-14 20:40): the long chat families (ifchat/multilingual/science, 83-93% of their assistant tokens
# in turns > 1k) took ~65% of the loss on capacity the prune removed, gains stopped at step 200 and the code families
# regressed (opencode .222->.244, kimi31b .377->.414, swe .294->.313).
#   corpus = sft12_build/<render> (sft12_select.py: agentic/opencode/hermes/kimi31b + a short-ifchat anchor ~10% of the
#            loss, no segment with > 1024 scored tokens, re-packed by the renderer's Packer); same 20 eval files.
#   init   = the 98e mlk8-noa bf16 (NOT the sft11-mlk8 checkpoints)
#   scope  = dense_mlp,router,routed_experts (attention frozen: the prune changed only experts/routing)
#   lr     = 2e-6 (was 4.74e-6), warmup 50 (was 10), one pass, cosine to min_lr 1e-7; eval every 100 (was 200)
#   stop rule: an EVAL where opencode, swe or cp is > init + 0.01, or kimi31b > init + 0.02 (owner 2026-11-15 06:29,
#            after step 100 stopped at kimi31b +0.0104; its eval is 11k tokens) -> touch PAUSE (checkpoint + exit) and
#            SFT12_STOP_RULE; the previous checkpoint is in the rolling keep-2 snapshots.
#   window gate: window_check's cut_end counts a window whose last token is <|tool_response> (50, a tool-call
#            segment filling the window exactly); this gate accepts 50 (sft12_select verified every segment ends on
#            <turn|> or <|tool_response>).
# Sentinels: SFT12_START / SFT12_RESUME / SFT12_EVAL / SFT12_STOP_RULE / SFT12_OK / SFT12_FAIL.
# Usage: sft12_train.sh <render dir name under sft12_build, e.g. render_oc33>
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft2_code_mlk8 CUDA_VISIBLE_DEVICES=1
NAME=${1:?usage: sft12_train.sh <render dir under sft12_build>}
M=/mnt/sdc/ml/v9a_sft11/mlk8-noa/gemma-4-20B-v9-agentic-sft11-mlk8-noa; CACHE=cache_teacher_r4
R=sft12_build/$NAME; RUN=runs/g4_bf16_sft12_98e_$NAME; LOG=train_sft12_$NAME.log; LR=2e-6
fail(){ echo "SFT12_FAIL $*" | tee -a $LOG; exit 1; }
S1=$RUN/stage1
[ -d "$R" ] || fail "no render dir $R"
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
$PY - $R <<'PY' || fail "window gate"
import glob, json, sys
bad = nobos = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.train.roles.jsonl")):
    for ln in open(f):
        ids = json.loads(ln)["ids"]
        nobos += ids[0] != 2
        bad += ids[-1] not in (0, 1, 106, 50)
print(f"[window gate] nobos {nobos} cut_end {bad}")
sys.exit(1 if bad or nobos else 0)
PY
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
[ "$N" -ge 200 ] || fail "only $N train windows"
STEPS=$(( N / 2 ))
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 150 ] || fail "only ${free}G free on /mnt/sdc (rolling 2 snapshots + final)"
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU1 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT12_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT12_START $(date -u +%FT%TZ) render $R windows $N steps $STEPS lr $LR train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:dense_mlp,router,routed_experts --bs 2 --steps $STEPS --lr $LR --fp-lr $LR --warmup 50 \
  --seq 16384 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-every 100 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm893-g4-98e-sft12-$NAME >> $LOG 2>&1 < /dev/null &
T=$!
echo $T > sft12_${NAME}_trainer.pid
mkdir -p $RUN/early
seen=0
while kill -0 $T 2>/dev/null; do
  mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then  # 94 GiB of 97.9: checkpoint and stop before an OOM
    echo "SFT12_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; touch $S1/PAUSE; paused=1
  fi
  n=$(grep -c '^EVAL ' $LOG)
  if [ "$n" -gt "$seen" ]; then
    seen=$n
    verdict=$($PY - $LOG <<'PY'
import json, sys
ev = [json.loads(l[5:]) for l in open(sys.argv[1]) if l.startswith("EVAL ")]
init, cur = ev[0]["families"], ev[-1]["families"]
fam = ["agentic.eval", "opencode.eval", "kimi31b.eval", "swe.eval", "cp.eval", "hermes.eval", "ifchat_v1.eval"]
row = " ".join(f"{f.split('.')[0]} {cur[f]['nll']:.3f}({cur[f]['nll'] - init[f]['nll']:+.3f})" for f in fam)
LIM = {"opencode.eval": 0.01, "kimi31b.eval": 0.02, "swe.eval": 0.01, "cp.eval": 0.01}  # kimi31b 0.02: owner 11-15 06:29
worse = [f for f, d in LIM.items() if cur[f]["nll"] > init[f]["nll"] + d]
print(("STOP " if worse and len(ev) > 1 else "OK ") + ev[-1]["tag"] + " " + row + (" worse:" + ",".join(worse) if worse else ""))
PY
)
    echo "SFT12_EVAL $(date -u +%FT%TZ) $verdict" | tee -a $LOG
    if [ "${verdict%% *}" = STOP ] && [ -z "${stopped:-}" ]; then
      echo "SFT12_STOP_RULE $(date -u +%FT%TZ) $verdict" | tee -a $LOG; touch $S1/PAUSE; stopped=1
    fi
  fi
  sleep 30
done
wait $T
[ -n "${stopped:-}" ] && { echo "SFT12_STOPPED_BY_RULE $(date -u +%FT%TZ) last $(cat $S1/last.json | tr -d '\n ')" | tee -a $LOG; exit 0; }
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT12_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
