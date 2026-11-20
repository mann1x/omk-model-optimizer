#!/bin/bash
# bm898_kfac_q6k_twin.sh <gpu> -- bm-898 step 3d (owner 2026-11-19 16:58 CET "once the factors are done, build the k-fac
# q6_k twin"): kfac_ladder_twins.sh's Q6_K tier for gemma-4-20B-v9-agentic-base-mlk8-noa-sft11, same commands as the
# sft12 twin (omk/models/gemma-4-20b-v9-agentic/quant/kfac_ladder_twins.sh 20b Q6_K all), own inputs:
#   K       = runs/g4_20b_base_kfac           (factors_w512_s2048_g10 + eval_20b_base.pt from bm898_kfac_inputs.sh)
#   BASE    = the sft11 F16 GGUF (eos 106)    IMAT = gguf/imatrix_ac.dat (AC, 9590 chunks)
#   ptq.gguf = a HARD LINK to the plain Q6_K bm898_quants.sh built (quantize_gguf 1081fc2 = llama-quantize b12386
#              --imatrix + the attention rule --tensor-type attn_(q|k|v|output)=q8_0: the same call the twins script
#              makes; no second 18 GB file, no re-quantize)
#   kfac.gguf = omk_qad.kfac_round --stems all --qtype auto --ref ptq.gguf (every MLP tensor re-rounded with the
#              factors at the same per-tensor types; attention / embd byte-copied)   -> identical size
#   score    = step0_gguf_student_eval.py (klk + top1) of both against eval_20b_base.pt -> LT_TWIN line
# Waits for KFI_DONE (kfac_inputs.chain.2.log). ptq.gguf is KEPT (no --rm-ptq; owner's word required to remove).
# Never overwrites. Sentinels LT_START / LT_PTQ / LT_KFAC / LT_TWIN / LT_DONE / LT_FAIL (same names as the twins).
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_kfac_q6k_twin.sh 0 > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/kfac_q6k_twin.chain.log 2>&1 < /dev/null &
set -uo pipefail
GPU=${1:?gpu index}; TIER=Q6_K
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11; GG=$O/gguf
K=/mnt/sdc/omk-model-optimizer/runs/g4_20b_base_kfac
PY=/srv/ml/envs/envs/omk-modelopt/bin/python
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
LLAMA=/mnt/sdc/ml/llama.cpp-b12386
TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
R26=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
BASE=$GG/$NAME-F16.gguf; IMAT=$GG/imatrix_ac.dat; EVAL=$K/eval_20b_base.pt
FACTORS=$K/factors_w512_s2048_g10; PTQ=$GG/$NAME-$TIER.gguf
KLOG=/mnt/sdc/omk-model-optimizer/runs/bm898_f16/kfac_inputs.chain.2.log
D=$K/ladder/$TIER; mkdir -p $D; LOG=$K/ladder/twins.log
say() { echo "[lt 20b-base $TIER $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "LT_FAIL $*"; exit 1; }
need_disk() { local a; a=$(df --output=avail -BG /mnt/sdc | tail -n 1 | tr -dc 0-9); [ "$a" -ge "$1" ] || die "/mnt/sdc ${a}G < $1G"; }
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f/%.4f' % (r['klk'], r['top1']))"; }
score_one() {  # <gguf> <json> <log>
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  $PY $R26/step0_gguf_student_eval.py --gguf $1 --eval $EVAL --bin $TOPK --omk $OMK --topk 256 --device cuda:0 \
    --out $2 > $3 2>&1 < /dev/null || { tail -n 12 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}

# ---- preflight on everything that does not depend on the factors ---------------------------------------------------
[ -s $BASE ] || die "no base $BASE"; [ -s $IMAT ] || die "no imatrix $IMAT"; [ -s $PTQ ] || die "no plain $TIER at $PTQ"
sha256sum -c $PTQ.sha256 > /dev/null || die "$PTQ sha mismatch vs $PTQ.sha256"
[ -s $R26/step0_gguf_student_eval.py ] || die "no $R26/step0_gguf_student_eval.py"; [ -x $TOPK ] || die "no $TOPK"
[ -s $OMK/omk_qad/kfac_round.py ] || die "no kfac_round at $OMK"
[ -e $D/kfac.gguf ] && die "$D/kfac.gguf exists (never overwritten)"
say "LT_START stage all base $BASE imatrix $IMAT factors $FACTORS ptq $PTQ (waiting for KFI_DONE in $KLOG)"

# ---- wait for the factors + eval reference --------------------------------------------------------------------------
t=0
until grep -qE "KFI_DONE|KFI_FAIL" $KLOG 2>/dev/null; do
  [ $t -ge 14400 ] && die "K-FAC inputs chain not done after 4 h ($KLOG)"
  sleep 60; t=$((t + 60))
done
grep -q KFI_DONE $KLOG || die "K-FAC inputs chain failed: $(grep KFI_FAIL $KLOG | tail -n 1 | cut -c1-200)"
n=$(ls $FACTORS/layer_*.pt 2>/dev/null | wc -l); [ "$n" -eq 30 ] || die "factors incomplete: $n/30 layer files"
[ -s $EVAL ] || die "no eval reference $EVAL"
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free ($used MiB)"

# ---- ptq: hard link to the quantize_gguf tier ----------------------------------------------------------------------
if [ -s $D/ptq.gguf ]; then say "ptq.gguf exists, kept"; else
  ln $PTQ $D/ptq.gguf || die "ln $PTQ"
  cp $PTQ.sha256 $D/ptq.gguf.sha256
  say "LT_PTQ hard link of $PTQ ($(du -h $D/ptq.gguf | cut -f1), sha $(cut -c1-16 $D/ptq.gguf.sha256)) attention q8_0 (quantize_gguf 1081fc2 rule)"
fi

# ---- kfac --------------------------------------------------------------------------------------------------------------
need_disk 40
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU OMK_GGML_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  $PY -m omk_qad.kfac_round --factors $FACTORS --base $BASE --imatrix $IMAT \
  --qtype auto --stems all --ref $D/ptq.gguf --out $D/kfac.gguf --device cuda:0 --llama-cpp $LLAMA \
  --report $D/kfac.json > $D/kfac.log 2>&1 < /dev/null
grep -q KFACR_DONE $D/kfac.log || { tail -n 6 $D/kfac.log | tee -a $LOG; die "kfac_round"; }
say "encoder mismatches: $(grep -c KFACR_ENCODER_MISMATCH $D/kfac.log) (expected 0: rtn == llama-quantize bytes)"
sha256sum $D/kfac.gguf > $D/kfac.gguf.sha256
say "LT_KFAC $(( $(date +%s) - t0 ))s $(grep KFACR_DONE $D/kfac.log | cut -c1-120)"
grep KFACR_STEM $D/kfac.log | tee -a $LOG

# ---- score both -----------------------------------------------------------------------------------------------------
for arm in ptq kfac; do
  [ -s $D/$arm.eval.json ] && continue
  score_one $D/$arm.gguf $D/$arm.eval.json $D/$arm.eval.log || die "score $arm"
done
SZ=$(stat -c %s $D/ptq.gguf); SZK=$(stat -c %s $D/kfac.gguf)
say "LT_TWIN $TIER size $(( SZ / 1048576 ))MiB kfac $(( SZK / 1048576 ))MiB  llama.cpp $(klk_of $D/ptq.eval.json)  K-FAC $(klk_of $D/kfac.eval.json)"
say "LT_DONE stage all (ptq.gguf kept: hard link of the plain tier)"
