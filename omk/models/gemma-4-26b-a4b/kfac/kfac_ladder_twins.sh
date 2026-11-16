#!/usr/bin/env bash
# bm-892/893/895: K-FAC vs llama.cpp imatrix, tier by tier, same layout, same size — the publish decision.
#
#   kfac_ladder_twins.sh <26b|20b> <TIER> <stage: ptq|kfac|score|all> [--rm-ptq]
#
# For a tier T of the published ladder:
#   ptq   : llama-quantize T from the base GGUF with the imatrix and the Gemma-4 A4B attention rule
#           (below Q4_K_M attention Q6_K, Q4_K_M and above Q8_0 — the same --tensor-type quantize_gguf emits)
#             -> ladder/T/ptq.gguf                       (what quantize_gguf would publish today)
#   kfac  : omk_qad.kfac_round --stems all --qtype auto --ref ptq.gguf: every MLP tensor (routed gate_up + down,
#           dense gate/up/down) re-rounded with the K-FAC factors at the SAME per-tensor types, attention/embd kept
#             -> ladder/T/kfac.gguf                      (identical size; rounding-only difference)
#   score : both files against the model's own bf16 reference eval set (step0_gguf_student_eval, klk + top1)
#             -> ladder/T/{ptq,kfac}.eval.json, a TWIN line in ladder/twins.log
#   --rm-ptq: after BOTH scores are recorded, remove ptq.gguf (owner's word required; default keep).
# Sentinels: LT_START / LT_PTQ / LT_KFAC / LT_TWIN / LT_DONE / LT_FAIL. Never overwrites an existing file.
set -u
WHICH=${1:?26b|20b}; TIER=${2:?tier}; STAGE=${3:?ptq|kfac|score|all}; RMPTQ=0; [ "${4:-}" = "--rm-ptq" ] && RMPTQ=1
PY=/srv/ml/envs/envs/omk-modelopt/bin/python
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
LLAMA=/mnt/sdc/ml/llama.cpp-b12386
QUANT=$LLAMA/build/bin/llama-quantize
TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
R26=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
case $WHICH in
  26b) K=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac
       BASE=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal-BF16-eos106.gguf
       IMAT=$R26/cache/imatrix.gguf; EVAL=$R26/cache/eval.pt ;;
  20b) K=/mnt/sdc/omk-model-optimizer/runs/g4_20b_kfac
       BASE=/srv/ml/scratch/sft12_ac_imatrix/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12-F16.gguf
       IMAT=/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12/gguf/imatrix_ac.dat; EVAL=$K/eval_20b.pt ;;
  *) echo "LT_FAIL which=$WHICH"; exit 1 ;;
esac
FACTORS=$K/factors_w512_s2048_g10
D=$K/ladder/$TIER; mkdir -p $D; LOG=$K/ladder/twins.log
# owner 2026-11-16 13:11 "delete the scored twins": a marker file carries the grant to every tier the driver runs
[ -e $K/ladder/.rm_ptq_granted ] && RMPTQ=1
say() { echo "[lt $WHICH $TIER $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "LT_FAIL $*"; exit 1; }
# the attention rule (quantize_gguf.gemma4_attention_type): Q4_K_M and above -> Q8_0, else Q6_K; float tiers none
attn_type() {
  case $1 in
    Q4_K_M|Q4_K_L|Q5_K_S|Q5_K_M|Q5_K_L|Q5_0|Q5_1|Q6_K|Q6_K_L|Q8_0) echo q8_0 ;;
    F16|BF16|F32) echo "" ;;
    *) echo q6_k ;;
  esac
}
need_disk() { local a; a=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9); [ "$a" -ge "$1" ] || die "/mnt/sdc ${a}G < $1G"; }
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f/%.4f' % (r['klk'], r['top1']))"; }
free_gpu() {  # index of a GPU with < 60000 MiB in use (the QAD cache fills GPU0, its trainer GPU1); waits up to 3 h
  local i u
  for _ in $(seq 1 540); do
    while read -r i u; do [ "$u" -lt 60000 ] && { echo "$i"; return 0; }; done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ,)
    sleep 20
  done
  return 1
}
score_one() {  # <gguf> <json> <log>  on a free GPU (bug-951: a hard-wired GPU0 failed "llama_topk exited before it was ready" while the 20B QAD cache held it)
  local g; g=$(free_gpu) || { echo "no GPU below 60000 MiB in 3 h" | tee -a $LOG; return 1; }
  say "scoring $(basename $1) on GPU$g"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  $PY $R26/step0_gguf_student_eval.py --gguf $1 --eval $EVAL --bin $TOPK --omk $OMK --topk 256 --device cuda:0 \
    --out $2 > $3 2>&1 < /dev/null || { tail -12 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}

say "LT_START stage $STAGE base $BASE imatrix $IMAT factors $FACTORS"
[ -s $BASE ] || die "no base $BASE"; [ -s $IMAT ] || die "no imatrix $IMAT"

if [ $STAGE = ptq ] || [ $STAGE = all ]; then
  if [ -s $D/ptq.gguf ]; then say "ptq.gguf exists, kept"; else
    need_disk 60
    AT=$(attn_type $TIER); ARGS=(--imatrix $IMAT); [ -n "$AT" ] && ARGS+=(--tensor-type "blk\.\d+\.attn_(q|k|v|output)\.weight=$AT")
    $QUANT --dry-run "${ARGS[@]}" $BASE $D/ptq.gguf $TIER > $D/ptq.dry_run.log 2>&1 < /dev/null || die "ptq dry-run: $(tail -3 $D/ptq.dry_run.log | tr '\n' ' ')"
    t0=$(date +%s)
    # 8 threads: the seeded QAD trainer on GPU1 encodes with libggml on the CPU and halves its speed when starved
    $QUANT "${ARGS[@]}" $BASE $D/ptq.partial.gguf $TIER 8 > $D/ptq.log 2>&1 < /dev/null || { tail -5 $D/ptq.log | tee -a $LOG; die "ptq"; }
    mv $D/ptq.partial.gguf $D/ptq.gguf || die mv
    sha256sum $D/ptq.gguf > $D/ptq.gguf.sha256
    say "LT_PTQ $(du -h $D/ptq.gguf | cut -f1) $(( $(date +%s) - t0 ))s attention ${AT:-float}; $(grep -c 'falling back' $D/ptq.log) fallbacks"
  fi
fi

if [ $STAGE = kfac ] || [ $STAGE = all ]; then
  # a kfac.gguf is complete only when its rounder logged KFACR_DONE: kfac_round writes --out as a byte copy of --ref
  # FIRST and replaces the payloads layer by layer, so a killed rounder leaves a file that scores like the PTQ
  # (bug-949: 20B Q5_K_S "K-FAC 0.0054 vs 0.0054" was such a copy). A stale file is quarantined, never deleted.
  if [ -s $D/kfac.gguf ] && ! grep -q KFACR_DONE $D/kfac.log 2>/dev/null; then
    S=$(date -u +%Y%m%dT%H%MZ); mv $D/kfac.gguf $D/kfac.gguf.partial_$S; [ -e $D/kfac.log ] && mv $D/kfac.log $D/kfac.log.partial_$S
    say "kfac.gguf had no KFACR_DONE: quarantined as kfac.gguf.partial_$S, re-rounding"
  fi
  if [ -s $D/kfac.gguf ]; then say "kfac.gguf exists, kept (KFACR_DONE)"; else
    [ -s $D/ptq.gguf ] || die "no ptq.gguf (run ptq first)"
    ls $FACTORS/layer_29.pt >/dev/null 2>&1 || die "factors incomplete at $FACTORS"
    need_disk 40
    t0=$(date +%s)
    # 8 libggml encoder threads: the QAD trainer on GPU1 shares the 32 cores (its refresh encodes on the CPU too).
    # CUDA_HOME/PATH = cuda-12.8: the IQ3/IQ4 CUDA encoders are torch extensions; /usr/local/cuda is 13.3 on bs2 and its
    # nvcc cannot compile torch 2.x headers (26B IQ4_XS twin failed 2026-11-16 17:31Z, bug-947)
    CUDA_VISIBLE_DEVICES=0 OMK_GGML_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH $PY -m omk_qad.kfac_round --factors $FACTORS --base $BASE --imatrix $IMAT \
      --qtype auto --stems all --ref $D/ptq.gguf --out $D/kfac.gguf --device cuda:0 --llama-cpp $LLAMA \
      --report $D/kfac.json > $D/kfac.log 2>&1 < /dev/null
    grep -q KFACR_DONE $D/kfac.log || { tail -6 $D/kfac.log | tee -a $LOG; die "kfac_round"; }
    say "encoder mismatches: $(grep -c KFACR_ENCODER_MISMATCH $D/kfac.log) (expected 0: rtn == llama-quantize bytes)"
    sha256sum $D/kfac.gguf > $D/kfac.gguf.sha256
    say "LT_KFAC $(( $(date +%s) - t0 ))s $(grep KFACR_DONE $D/kfac.log | cut -c1-120)"
    grep KFACR_STEM $D/kfac.log | tee -a $LOG
  fi
fi

if [ $STAGE = score ] || [ $STAGE = all ]; then
  [ -s $EVAL ] || die "no eval reference $EVAL"
  for arm in ptq kfac; do
    [ -s $D/$arm.gguf ] || die "no $arm.gguf"
    [ -s $D/$arm.eval.json ] && continue
    score_one $D/$arm.gguf $D/$arm.eval.json $D/$arm.eval.log || die "score $arm"
  done
  SZ=$(stat -c %s $D/ptq.gguf); SZK=$(stat -c %s $D/kfac.gguf)
  say "LT_TWIN $TIER size $(( SZ / 1048576 ))MiB kfac $(( SZK / 1048576 ))MiB  llama.cpp $(klk_of $D/ptq.eval.json)  K-FAC $(klk_of $D/kfac.eval.json)"
  if [ $RMPTQ = 1 ]; then rm -v $D/ptq.gguf | tee -a $LOG; fi
fi
say "LT_DONE stage $STAGE"
