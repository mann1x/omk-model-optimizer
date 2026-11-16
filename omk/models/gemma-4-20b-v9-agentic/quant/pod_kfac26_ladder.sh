#!/bin/bash
# pod_kfac26_ladder.sh -- RUNS ON THE EVAL POD (2x RTX 3090, 128 cores). The 26B agentic K-FAC ladder, tier by tier,
# the same twin recipe as bs2's kfac_ladder_twins.sh (ptq = llama-quantize b12386 + cache imatrix + Gemma-4 A4B
# attention rule; kfac = omk_qad.kfac_round --stems all --qtype auto --ref ptq.gguf; score = step0_gguf_student_eval
# top-256 KL against cache/eval.pt). Inputs staged by pod_kfac26_stage_from_bs2.sh under /workspace/kfac26; finished
# tiers are pulled back to bs2 by pod_kfac26_collect.sh (marker .tier_done here, .collected when bs2 has them), where
# the HF publisher / ollama pusher / solidpc archiver take over. Owner 2026-11-16: 26B lane stopped on bs2 ("use bs2
# for the qad"), ladders continue on the pod. Waits for the staged inputs (inputs.sha written by bs2) and for both GPUs
# to be released by the Q6_K suite (no llama-server, < 1000 MiB used, 3 consecutive minutes). GPU0 rounds, GPU1 scores.
# ptq.gguf is removed after both scores (owner 13:11 "delete the scored twins", marker .rm_ptq_granted on bs2).
# usage: pod_kfac26_ladder.sh [TIER ...]      sentinels PL_START / PL_WAIT / PL_STEP / LT_PTQ / LT_KFAC / LT_TWIN /
#        PL_TIER_DONE / PL_TIER_FAIL / PL_DONE in /workspace/kfac26/ladder/twins.log
set -u
D=/workspace/kfac26; LLAMA=/workspace/llama.cpp; ENV=/workspace/miniconda/envs/omnimergekit; PY=$ENV/bin/python
OMK=$D/code/omk-wip-iq2w; TOPK=$OMK/omk_qad/llama_topk/llama_topk
BASE=$D/base.gguf; IMAT=$D/cache/imatrix.gguf; EVAL=$D/cache/eval.pt; FACTORS=$D/factors_w512_s2048_g10
STEP0=$D/cache/step0_gguf_student_eval.py; QUANT=$LLAMA/build/bin/llama-quantize
LOG=$D/ladder/twins.log; mkdir -p $D/ladder
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
if [ $# -gt 0 ]; then TIERS=("$@"); else TIERS=(Q5_K_S Q4_K_S Q6_K IQ4_NL Q3_K_M IQ3_XXS IQ3_M IQ3_XS Q8_0 IQ2_M IQ2_S IQ2_XS); fi
say() { echo "[pl $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
attn_type() {  # quantize_gguf.gemma4_attention_type: Q4_K_M and above -> Q8_0, else Q6_K
  case $1 in Q4_K_M|Q4_K_L|Q5_K_S|Q5_K_M|Q5_K_L|Q5_0|Q5_1|Q6_K|Q6_K_L|Q8_0) echo q8_0 ;; F16|BF16|F32) echo "" ;; *) echo q6_k ;; esac
}
need_disk() {  # waits up to 3 h for the bs2 collector to pull finished tiers off the pod
  local a; for _ in $(seq 1 180); do a=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9); [ "$a" -ge "$1" ] && return 0; sleep 60; done; return 1
}
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f/%.4f' % (r['klk'], r['top1']))"; }
gpus_free() {  # no llama-server and both GPUs below 1000 MiB
  ps -eo args | grep -q "[l]lama-server" && return 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ if ($1+0 >= 1000) bad=1 } END { exit bad }'
}
say "PL_START tiers: ${TIERS[*]}"
# --- inputs (staged by bs2; inputs.sha = "<sha256> <bytes>" of the base) ---
for _ in $(seq 1 720); do [ -s $D/inputs.sha ] && break; sleep 60; done
[ -s $D/inputs.sha ] || { say "PL_FAIL inputs.sha never arrived"; exit 1; }
read -r WANT BYTES < $D/inputs.sha
# the stage rsyncs (factors, base) may still be running: wait until the base has its full size and layer_29 exists
for _ in $(seq 1 720); do [ "$(stat -c %s $BASE 2>/dev/null)" = "$BYTES" ] && [ -e $FACTORS/layer_29.pt ] && break; sleep 60; done
if [ ! -e $D/.inputs_ok ]; then
  [ "$(stat -c %s $BASE)" = "$BYTES" ] || { say "PL_FAIL base size $(stat -c %s $BASE) != $BYTES after 12 h"; exit 1; }
  sleep 120  # let the factors rsync settle (layer_29 is written last, but --inplace files may still be open)
  got=$(sha256sum $BASE | cut -d' ' -f1); [ "$got" = "$WANT" ] || { say "PL_FAIL base sha $got != $WANT"; exit 1; }
  n=$(ls $FACTORS | wc -l); say "inputs ok: base sha $WANT, $n factor files"; touch $D/.inputs_ok
fi
# --- llama_topk against the pod's b12386 build (sm_86) ---
if [ ! -x $TOPK ]; then
  g++ -O3 -fopenmp -std=c++17 $OMK/omk_qad/llama_topk/llama_topk.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
      -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o $TOPK > $D/ladder/llama_topk.build.log 2>&1 \
    || { say "PL_FAIL llama_topk build: $(tail -3 $D/ladder/llama_topk.build.log | tr '\n' ' ')"; exit 1; }
  say "llama_topk built"
fi
[ -x $QUANT ] || { say "PL_FAIL no $QUANT"; exit 1; }
# --- GPUs: wait for the Q6_K suite ---
ok=0; while :; do if gpus_free; then ok=$((ok + 1)); [ $ok -ge 3 ] && break; else ok=0; say "PL_WAIT GPUs busy (suite): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 540; fi; sleep 60; done
say "GPUs free, ladder starts"
score_one() {  # <gguf> <json> <log>  GPU1
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$OMK LD_LIBRARY_PATH=$LLAMA/build/bin:${LD_LIBRARY_PATH:-} \
  $PY $STEP0 --gguf $1 --eval $EVAL --bin $TOPK --omk $OMK --topk 256 --device cuda:0 --out $2 > $3 2>&1 < /dev/null \
    || { tail -12 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}
one_tier() {
  local T=$1 TD=$D/ladder/$1 t0 AT ARGS
  mkdir -p $TD
  [ -e $TD/.tier_done ] && { say "$T: done already"; return 0; }
  say "PL_STEP $T ($(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)G free)"
  if [ ! -s $TD/ptq.gguf ]; then
    need_disk 70 || { say "PL_TIER_FAIL $T: disk"; return 1; }
    AT=$(attn_type $T); ARGS=(--imatrix $IMAT); [ -n "$AT" ] && ARGS+=(--tensor-type "blk\.\d+\.attn_(q|k|v|output)\.weight=$AT")
    $QUANT --dry-run "${ARGS[@]}" $BASE $TD/ptq.gguf $T > $TD/ptq.dry_run.log 2>&1 < /dev/null || { say "PL_TIER_FAIL $T ptq dry-run: $(tail -2 $TD/ptq.dry_run.log | tr '\n' ' ')"; return 1; }
    t0=$(date +%s)
    $QUANT "${ARGS[@]}" $BASE $TD/ptq.gguf $T 64 > $TD/ptq.log 2>&1 < /dev/null || { say "PL_TIER_FAIL $T ptq: $(tail -2 $TD/ptq.log | tr '\n' ' ')"; rm -f $TD/ptq.gguf; return 1; }
    sha256sum $TD/ptq.gguf > $TD/ptq.gguf.sha256
    say "LT_PTQ $T $(du -h $TD/ptq.gguf | cut -f1) $(( $(date +%s) - t0 ))s attention ${AT:-none}"
  fi
  if [ -s $TD/kfac.gguf ] && ! grep -q KFACR_DONE $TD/kfac.log 2>/dev/null; then
    S=$(date -u +%Y%m%dT%H%MZ); mv $TD/kfac.gguf $TD/kfac.gguf.partial_$S; [ -e $TD/kfac.log ] && mv $TD/kfac.log $TD/kfac.log.partial_$S
    say "$T: kfac.gguf without KFACR_DONE quarantined as kfac.gguf.partial_$S"
  fi
  if [ ! -s $TD/kfac.gguf ]; then
    need_disk 40 || { say "PL_TIER_FAIL $T: disk"; return 1; }
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 OMK_GGML_THREADS=32 OMP_NUM_THREADS=32 PYTHONPATH=$OMK LD_LIBRARY_PATH=$LLAMA/build/bin:${LD_LIBRARY_PATH:-} \
      $PY -m omk_qad.kfac_round --factors $FACTORS --base $BASE --imatrix $IMAT --qtype auto --stems all \
      --ref $TD/ptq.gguf --out $TD/kfac.gguf --device cuda:0 --llama-cpp $LLAMA --report $TD/kfac.json > $TD/kfac.log 2>&1 < /dev/null
    grep -q KFACR_DONE $TD/kfac.log || { tail -6 $TD/kfac.log | tee -a $LOG; say "PL_TIER_FAIL $T kfac_round"; return 1; }
    sha256sum $TD/kfac.gguf > $TD/kfac.gguf.sha256
    say "LT_KFAC $T $(( $(date +%s) - t0 ))s $(grep -o 'KFACR_DONE.*' $TD/kfac.log | head -1 | cut -c1-120) encoder mismatches $(grep -c KFACR_ENCODER_MISMATCH $TD/kfac.log)"
  fi
  for arm in ptq kfac; do
    [ -s $TD/$arm.eval.json ] && continue
    score_one $TD/$arm.gguf $TD/$arm.eval.json $TD/$arm.eval.log || { say "PL_TIER_FAIL $T score $arm"; return 1; }
  done
  say "LT_TWIN $T size $(( $(stat -c %s $TD/ptq.gguf) / 1048576 ))MiB kfac $(( $(stat -c %s $TD/kfac.gguf) / 1048576 ))MiB  llama.cpp $(klk_of $TD/ptq.eval.json)  K-FAC $(klk_of $TD/kfac.eval.json)"
  rm -f $TD/ptq.gguf && say "$T: ptq.gguf removed (scored twin, owner grant 13:11)"
  touch $TD/.tier_done; say "PL_TIER_DONE $T"
}
fail=0
for T in "${TIERS[@]}"; do one_tier $T || fail=$((fail + 1)); done
say "PL_DONE ${#TIERS[@]} tiers, $fail failed"
