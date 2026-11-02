#!/bin/bash
# bm-889 sft5 (owner 2026-11-02 16:44 CET "when the render is done, stop sft4 and launch sft5 with it"): sft4's trainer,
# recipe and LR rule on sft4_build/render_32k_v3_mlast2 = the sft4 corpus re-rendered with chat_template_mlast2.jinja
# (multi-line call arguments last, nested objects too: a value with a newline is followed only by another multi-line value or a closing bracket; values unchanged).
# Sentinels: SFT5_HOLD / SFT5_START / SFT5_RESUME / SFT5_OK / SFT5_FAIL. Resumable: rerun the same line after a crash.
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export PYTHONPATH=/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft4_code CUDA_VISIBLE_DEVICES=0
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it; CACHE=cache_teacher_r4; RUN=runs/g4_bf16_sft5; R=sft4_build/render_32k_v3_mlast2; LOG=train_sft5.log
fail(){ echo "SFT5_FAIL $*" | tee -a $LOG; exit 1; }
while [ -e sft5_train.HOLD ]; do [ -z "${held:-}" ] && echo "SFT5_HOLD $(date -u +%FT%TZ)"; held=1; sleep 30; done
S1=$RUN/stage1
[ -e $S1/final.pt ] && fail "$RUN already complete"
[ -e "$RUN" ] && [ ! -e $S1/last.pt ] && fail "$RUN exists without last.pt"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY="$(grep -oE 'WANDB_API_KEY=[^ ;]+' /root/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")"
[ -n "${WANDB_API_KEY:-}" ] || fail "WANDB_API_KEY not found"
$PY -c "import wandb" || fail "wandb does not import"
[ -e $R/OVERLAP_OK ] || fail "$R has no OVERLAP_OK"
[ -s $R/CORPUS_MANIFEST.json ] || fail "$R has no CORPUS_MANIFEST.json"
$PY - "$R" <<'PYC' || fail "manifest check"
import json,hashlib,sys,os
d=sys.argv[1]; m=json.load(open(d+'/CORPUS_MANIFEST.json'))
for f,v in m['files'].items():
    h=hashlib.sha256(open(os.path.join(d,f),'rb').read()).hexdigest()
    assert h==v['sha256'], f'{f} sha256 differs from the manifest'
print('MANIFEST_OK',len(m['files']))
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
[ "$free" -ge 300 ] || fail "only ${free}G free on /mnt/sdc"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || fail "GPU0 not free ($used MiB used)"
if [ -e "$RUN" ]; then echo "SFT5_RESUME $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR" | tee -a $LOG
else echo "SFT5_START $(date -u +%FT%TZ) windows $N steps $STEPS lr $LR train $(echo $DATA | tr , '\n' | wc -l) eval $(echo $EVAL | tr , '\n' | wc -l)" | tee -a $LOG; fi
$PY -m omk_qad.train --phase train --mode bf16 --bf16-latent fp16 --model $M --cache $CACHE --out $RUN \
  --train-scope classes:attention,dense_mlp,router,routed_experts --bs 1 --steps $STEPS --lr $LR --fp-lr $LR --warmup 10 \
  --seq 32768 --loss ce --sft-data "$DATA" --sft-eval "$EVAL" --mtp freeze --eval-set heldout --eval-every 200 \
  --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 --wandb-project omk-qad \
  --wandb-name bm889-g4-bf16-sft5 >> $LOG 2>&1 < /dev/null &
T=$!
mkdir -p $RUN/early
while kill -0 $T 2>/dev/null; do
  for s in 0000100 0000200 0000400; do [ -e $S1/ckpt_step$s.pt ] && [ ! -e $RUN/early/ckpt_step$s.pt ] && ln $S1/ckpt_step$s.pt $RUN/early/; done
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${mem:-0}" -gt 96250 ] && [ -z "${paused:-}" ]; then
    echo "SFT5_MEM_PAUSE $(date -u +%FT%TZ) ${mem} MiB" | tee -a $LOG; echo pause > $S1/control; paused=1
  fi
  sleep 30
done
wait $T
grep -q "QAD_DONE" $LOG || fail "train: $(tail -3 $LOG | tr "\n" " " | cut -c1-300)"
echo "SFT5_OK $(date -u +%FT%TZ) steps $STEPS" | tee -a $LOG
