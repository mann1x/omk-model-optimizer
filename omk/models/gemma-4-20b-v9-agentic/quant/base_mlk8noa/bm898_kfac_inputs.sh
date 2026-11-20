#!/bin/bash
# bm898_kfac_inputs.sh <gpu> -- bm-898 step 3c (owner 2026-11-19 16:45 CET "right q6_k is k-fac, once done with the quant
# let's start with collecting the factors"): the K-FAC inputs of gemma-4-20B-v9-agentic-base-mlk8-noa-sft11, the SAME
# recipe as the sft12 20B (kfac_collect_20b.sh / build_eval_from_gguf.py, omk/models/gemma-4-20b-v9-agentic/quant):
#   1. wait for the PTQ tiers (bm898_quants.sh: B898Q_DONE in quants.chain.log; B898Q_FAIL aborts) -- GPU0 is shared
#      with nothing else under claim 52cc64e8d7bd;
#   2. factors: omk_qad.kfac --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10
#      on the sft11 bf16 HF export, token windows from the 26B QAD cache (same tokenizer; labels sampled from this
#      model = true Fisher) -> $R/factors_w512_s2048_g10 (~59 GB, sft12 took 76 min on this GPU);
#   3. eval reference: the 26B cache's 44 held-out role rows with THIS model's F16 GGUF top-64 (llama_topk b12386)
#      -> $R/eval_20b_base.pt (sft12: 256,853 scored positions, ~5 min).
# Then (not here, owner's word): kfac_ladder_twins-style Q6_K twin = PTQ Q6_K (built) + kfac_round --stems all --qtype
# auto --ref <ptq Q6_K> + both scored on eval_20b_base.pt.
# Never overwrites. Sentinels KFI_START / KFI_STEP_OK <step> / KFI_DONE / KFI_FAIL.
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_kfac_inputs.sh 0 > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/kfac_inputs.chain.log 2>&1 < /dev/null &
set -uo pipefail
GPU=${1:?gpu index}
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11; M=$O/$NAME; GG=$O/gguf
R=/mnt/sdc/omk-model-optimizer/runs/g4_20b_base_kfac; mkdir -p $R
R26=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
PY=/srv/ml/envs/envs/omk-modelopt/bin/python; OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
QLOG=/mnt/sdc/omk-model-optimizer/runs/bm898_f16/quants.chain.log
OUT=$R/factors_w512_s2048_g10; EV=$R/eval_20b_base.pt
LOG=$R/kfac_inputs.driver.log
say() { echo "[kfi $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "KFI_FAIL $*"; exit 1; }
export PYTHONPATH=$OMK CUDA_VISIBLE_DEVICES=$GPU

# ---- preflight (everything the two steps need, before waiting) -----------------------------------------------------
[ -s $M/config.json ] && [ -e $M/BUILD_OK ] || die "no model at $M (config.json + BUILD_OK)"
[ -s $GG/$NAME-F16.gguf ] || die "no F16 $GG/$NAME-F16.gguf"
[ -s $R26/cache/eval.pt ] && [ -s $R26/cache/cache.json ] || die "26B cache missing at $R26/cache"
[ -s $OMK/omk_qad/kfac.py ] || die "no omk_qad.kfac at $OMK"
[ -x $TOPK ] || die "no llama_topk at $TOPK"
[ -s $R/build_eval_from_gguf.py ] || die "no $R/build_eval_from_gguf.py (rsync it from omk/models/gemma-4-20b-v9-agentic/quant/)"
[ -e $OUT ] && die "$OUT exists (never overwritten)"
[ -e $EV ] && die "$EV exists (never overwritten)"
say "KFI_START gpu $GPU model $M F16 $(cut -c1-16 $GG/$NAME-F16.gguf.sha256) out $R"

# ---- 1. wait for the PTQ tiers ---------------------------------------------------------------------------------------
t=0
until grep -qE "B898Q_DONE|B898Q_FAIL" $QLOG 2>/dev/null; do
  [ $t -ge 5400 ] && die "quant chain not done after 90 min ($QLOG)"
  sleep 30; t=$((t + 30))
done
grep -q B898Q_DONE $QLOG || die "quant chain failed: $(grep B898Q_FAIL $QLOG | tail -n 1 | cut -c1-200)"
say "KFI_STEP_OK quants ($(grep -c B898Q_TIER_OK $QLOG) tiers)"
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free ($used MiB)"
free=$(df --output=avail -BG /mnt/sdc | tail -n 1 | tr -dc 0-9); [ "${free:-0}" -ge 130 ] || die "only ${free}G free on /mnt/sdc (need 130G: factors ~59G + headroom)"

# ---- 2. factors (sft12 recipe verbatim) -----------------------------------------------------------------------------
say "factors -> $OUT (free ${free}G)"
$PY -m omk_qad.kfac --model $M \
  --cache $R26/cache --out $OUT --device cuda:0 \
  --windows 512 --seq 2048 --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10 \
  > $R/collect_g10.log 2>&1 < /dev/null || { tail -n 5 $R/collect_g10.log | cut -c1-300 | tee -a $LOG; die "kfac"; }
grep -q "KFAC_DONE" $R/collect_g10.log || die "no KFAC_DONE in collect_g10.log"
n=$(ls $OUT/layer_*.pt 2>/dev/null | wc -l); [ "$n" -eq 30 ] || die "expected 30 layer files in $OUT, found $n"
say "KFI_STEP_OK factors $OUT $(du -sh $OUT | cut -f1) $n layers"

# ---- 3. eval reference from this model's own F16 ---------------------------------------------------------------------
$PY $R/build_eval_from_gguf.py --gguf $GG/$NAME-F16.gguf --rows $R26/cache/eval.pt --out $EV \
  --bin $TOPK --omk $OMK --topk 64 --device cuda:0 > $R/eval_20b_base.log 2>&1 < /dev/null \
  || { tail -n 5 $R/eval_20b_base.log | cut -c1-300 | tee -a $LOG; die "eval reference"; }
grep -q "EVAL_FROM_GGUF OK" $R/eval_20b_base.log || die "no EVAL_FROM_GGUF OK"
say "KFI_STEP_OK eval $(grep 'EVAL_FROM_GGUF OK' $R/eval_20b_base.log | cut -c1-160)"
sha256sum $EV | tee $EV.sha256 | tee -a $LOG
say "KFI_DONE factors $OUT eval $EV"
