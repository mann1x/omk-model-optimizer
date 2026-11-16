#!/bin/bash
# bm-892/bm-896 seeded retrain, segment 3: resume the DRAINED run (step 301) to the full 600 steps.
# Owner 2026-11-16 18:13 CET: "resume it to 600, I want to be sure 300 steps are the most we can get. 16% is a lot and
# 3 hours of GPU time are well worth it." The plateau rule of the previous segments (< 0.001 klk per 100 steps -> drain)
# is OFF here: the run goes to --steps 600 (final.pt) unless the owner stops it.
#   1. wait for the previous chain (qad_seeded_E_chain_resume.sh) to finish exporting + scoring the step-301 checkpoint
#      (QSE_DONE / QSE_FAIL in its launcher.out) -- that GGUF (qadE6_kfacseed.gguf) is the "300 steps" artifact and is
#      never touched; this segment exports to qadE6_kfacseed_s600.gguf.
#   2. relaunch the ORIGINAL command verbatim (same code tree omk-wip-iq2cuda, --init-latents, --allow-change
#      ggml_encode); the trainer resets the control file to run on start (runctl.Control.__init__) and resumes last.pt.
#   3. wait for exit; export final.pt (or last.pt) -> qadE6_kfacseed_s600.gguf; score; compare with the step-301 score.
# Sentinels QS6_START / QS6_TRAIN / QS6_DONE / QS6_FAIL. Never overwrites a run or a GGUF. Usage: ...resume600.sh <gpu>
set -uo pipefail
GPU=${1:?gpu index}
R=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k; K=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac
PY=/srv/ml/envs/envs/omk-modelopt/bin/python; OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2cuda
LC=/mnt/sdc/ml/llama.cpp-b12386; MODEL=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
SEED=$K/round/E_latents  # --init-latents is part of the run identity: repeated verbatim, skipped on resume
LAY_TRAIN=$R/layout_E_attn_Q6_K/layout.pt
OUT=$R/qadE6_kfacseed; W=$R/seedgate_E; STEPS=${STEPS:-600}; REFRESH=${REFRESH:-4}
PREV=$R/qad_seeded_E_chain_resume.launcher.out
GG=$OUT/qadE6_kfacseed_s600.gguf
export PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
LOG=$W/chain.driver.log
say() { echo "[qs6 $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "QS6_FAIL $*"; exit 1; }
COMMON="--mode qad --model $MODEL --cache $R/cache --device cuda:0 --llama-cpp $LC
  --ggml-encode Q8_0 --imatrix $R/cache/imatrix.pt --imatrix-convention msq --train-scope gemma4
  --export-only embedding --latent-dtype fp16 --mtp train --topk 64"
score() {  # <gguf> <json> <log>
  $PY $R/step0_gguf_student_eval.py --gguf $1 --eval $R/cache/eval.pt --bin /mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk \
    --omk $OMK --topk 256 --device cuda:0 --out $2 > $3 2>&1 < /dev/null || { tail -15 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f %.4f' % (r['klk'], r['top1']))"; }

say "QS6_START waiting for the step-301 export+score of the previous chain ($PREV)"
for i in $(seq 1 240); do grep -q "QSE_DONE\|QSE_FAIL" $PREV && break; sleep 30; done
grep -q "QSE_DONE\|QSE_FAIL" $PREV || die "previous chain neither done nor failed after 2 h"
say "previous chain: $(grep "QSE_DONE\|QSE_FAIL" $PREV | tail -1 | cut -c1-200)"
sleep 20
# preconditions
[ -s $OUT/stage1/last.pt ] && [ -s $OUT/stage1/last.json ] || die "no last.pt/last.json in $OUT/stage1"
[ -e $OUT/stage1/final.pt ] && die "final.pt exists: the run finished, nothing to resume"
[ -e $GG ] && die "$GG exists (never overwritten)"
ls $OUT/stage1/last.pt.*.tmp 2>/dev/null | grep -q . && die "a checkpoint write is still in flight: $(ls $OUT/stage1/last.pt.*.tmp)"
ps -eo pid,args | grep "[o]mk_qad.train" | grep -q "$OUT" && die "a trainer/exporter on $OUT is still running: $(ps -eo pid,args | grep "[o]mk_qad.train" | grep "$OUT" | cut -c1-40)"
[ -s $OMK/omk_qad/iq2.py ] && [ -s $OMK/omk_qad/kernels/iq2.cu ] || die "no omk_qad.iq2 in $OMK"
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] || die "GPU$GPU busy ($used MiB)"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9); [ "${free:-0}" -ge 150 ] || die "only ${free}G free on /mnt/sdc (need 150: last.pt tmp + final)"
CKSTEP=$(python3 -c "import json; print(json.load(open('$OUT/stage1/last.json'))['step'])")
say "QS6_START gpu $GPU resume $OUT from last.pt step $CKSTEP -> $STEPS steps, plateau rule OFF (owner); code $OMK"
echo "===== RESUMED $(date -u +%FT%TZ) from step $CKSTEP to $STEPS (owner: no plateau stop) =====" >> $OUT.train.log
say "QS6_TRAIN $OUT: $STEPS steps, refresh $REFRESH, eval/save every 100, snapshots off, layout E + Q6_K attention"
$PY -m omk_qad.train --phase train $COMMON --layout $LAY_TRAIN --out $OUT --init-latents $SEED \
  --seq 4096 --n-seq 16000 --bs 1 --accum 1 --steps $STEPS --lr 2e-6 --fp-lr 2e-6 --min-lr 1e-7 --warmup 50 \
  --beta1 0 --beta2 0.999 --clip 1.0 --loss kl --anchor-as-teacher --mtp-weight 0.3 \
  --refresh $REFRESH --ggml-refresh $REFRESH --eval-every 100 --save-every 100 --snapshot-every 0 --log-every 10 \
  --allow-change ggml_encode \
  --wandb-project omk-qad-26b-agentic-iq2-qad --wandb-name qadE6_kfacseed_r$REFRESH >> $OUT.train.log 2>&1 < /dev/null &
TP=$!
sleep 420
kill -0 $TP 2>/dev/null || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer died in the first 7 min"; }
SEG=$(grep -n "===== RESUMED" $OUT.train.log | tail -1 | cut -d: -f1)
tail -n +$SEG $OUT.train.log > $OUT.train.resume600_head.txt
grep -q "loaded .*last.pt (step $CKSTEP)" $OUT.train.resume600_head.txt || { grep -i "RUN_REFUSED\|Traceback\|Error" $OUT.train.resume600_head.txt | head -5 | tee -a $LOG; die "resume did not load step $CKSTEP"; }
grep -q "encoders: libggml.*\['Q8_0'\]" $OUT.train.resume600_head.txt || die "libggml not restricted to Q8_0: $(grep encoders: $OUT.train.resume600_head.txt | head -2)"
say "trainer pid $TP alive; $(grep -m1 'loaded .*last.pt' $OUT.train.resume600_head.txt | cut -c1-120); $(grep -m1 'control file said' $OUT.train.resume600_head.txt | cut -c1-120)"
# no plateau watcher: run to --steps
while kill -0 $TP 2>/dev/null; do sleep 300; done
wait $TP; rc=$?
grep "^EVAL " $OUT.train.log | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l.split(' ', 1)[1]); print('  eval', r.get('tag', '?'), 'klk %.4f top1 %.4f' % (r['klk'], r['top1']))" | tee -a $LOG
CK=$OUT/stage1/final.pt; [ -s $CK ] || CK=$OUT/stage1/last.pt
[ -s $CK ] || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer exit $rc, no checkpoint"; }
FSTEP=$(python3 -c "import json,sys; f='$CK'.replace('.pt','.json'); print(json.load(open(f))['step'])" 2>/dev/null || echo "?")
say "trainer exit $rc; exporting $CK (step $FSTEP) -> $GG"
[ -e $GG ] && die "$GG appeared meanwhile (never overwritten)"
$PY -m omk_qad.train --phase export $COMMON --layout $LAY_TRAIN --out $OUT/export_run_s600 --load $CK --gguf $GG \
  > $OUT/export_s600.log 2>&1 < /dev/null || { tail -25 $OUT/export_s600.log | tee -a $LOG; die "final export"; }
sha256sum $GG | tee $GG.sha256 | tee -a $LOG
score $GG $OUT/qadE6_kfacseed_s600.eval.json $OUT/eval_s600.log || die "final score"
P300="n/a"; [ -s $OUT/qadE6_kfacseed.eval.json ] && P300=$(klk_of $OUT/qadE6_kfacseed.eval.json)
say "QS6_DONE seeded QAD step $FSTEP (Q6_K attn) $(klk_of $OUT/qadE6_kfacseed_s600.eval.json) | step-301 GGUF $P300 | K-FAC seed at Q6_K attn $(klk_of $W/seed_export_attnQ6K.json) | PTQ E 0.0675 | $(stat -c %s $GG | awk '{printf "%.2fGB", $1/1e9}') | checkpoint $CK"
