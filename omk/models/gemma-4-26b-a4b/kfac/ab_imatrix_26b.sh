#!/bin/bash
# bm-897 imatrix A/B for the published 26B agentic GGUF (owner 2026-11-26 18:37: "download the two i-matrix from HF and create two
# quants from the f16 that you can download from HF and compare the same quant"). The 26B K-FAC ladder used the QAD-cache imatrix
# (imatrix.gguf, 80e56b9b, 258 x 16k training windows); the 20B ladder the AtomicChat imatrix. Never compared until now.
# Inputs (all from HF, public): ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF {gemma-4-26B-A4B-agentic-F16.gguf, imatrix.dat (AC, 609a0a04),
#   imatrix.gguf (cache, 80e56b9b)}; ManniX-ITA/gemma-4-26B-A4B-agentic omk-artifacts/26b_agentic_iq2_r16k/cache/eval.pt (the teacher
#   reference the ladder tables were scored on).
# Per tier: llama-quantize b12386 F16 -> tier with --imatrix <AC|cache> + the Gemma-4 attention rule (Q8_0 from Q4_K_M up, Q6_K below),
#   the ONLY difference between the two arms is the imatrix; then the ladder twin score (step0_gguf_student_eval.py, llama_topk,
#   top-256 KL + top-1, GPU0). Results: $W/<tier>/<arm>.eval.json, summary $W/AB_RESULTS.tsv. Sentinels AB_* ; AB_DONE / AB_FAIL.
set -uo pipefail
W=/mnt/sdc/omk-model-optimizer/runs/imatrix_ab_26b; LOG=$W/ab.log; mkdir -p $W/hf
H=/srv/ml/envs/envs/omnimergekit/bin/hf; PY=/srv/ml/envs/envs/omk-modelopt/bin/python
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w; R26=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
QUANT=/mnt/sdc/ml/llama.cpp-b12386/build/bin/llama-quantize; TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
TIERS=${TIERS:-Q4_K_M}; ARMS=${ARMS:-cache ac}; RES=$W/AB_RESULTS_${ARMS// /_}.tsv
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
die(){ say "AB_FAIL $*"; exit 1; }
attn_type(){ case $1 in Q4_K_M|Q5_K_S|Q5_K_M|Q6_K|Q8_0) echo q8_0 ;; *) echo q6_k ;; esac; }
say "AB_START tiers [$TIERS]"
export HF_HUB_DISABLE_PROGRESS_BARS=1
$H download ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF gemma-4-26B-A4B-agentic-F16.gguf imatrix.dat imatrix.gguf --local-dir $W/hf >> $LOG 2>&1 < /dev/null || die "hf download GGUF repo"
$H download ManniX-ITA/gemma-4-26B-A4B-agentic omk-artifacts/26b_agentic_iq2_r16k/cache/eval.pt --local-dir $W/hf >> $LOG 2>&1 < /dev/null || die "hf download eval.pt"
F16=$W/hf/gemma-4-26B-A4B-agentic-F16.gguf; EVAL=$W/hf/omk-artifacts/26b_agentic_iq2_r16k/cache/eval.pt
declare -A IM=([ac]=$W/hf/imatrix.dat [cache]=$W/hf/imatrix.gguf [both]=$W/hf/imatrix.both.gguf)
# both = llama-imatrix --in-file imatrix.dat --in-file imatrix.gguf (no prompt: sums Σx² and counts = token-weighted mean, 4.91M AC + 4.23M cache tokens)
if [[ " $ARMS " == *" both "* ]] && [ ! -s $W/hf/imatrix.both.gguf ]; then
  /mnt/sdc/ml/llama.cpp-b12386/build/bin/llama-imatrix -m $W/hf/gemma-4-26B-A4B-agentic-F16.gguf --in-file $W/hf/imatrix.dat --in-file $W/hf/imatrix.gguf -o $W/hf/imatrix.both.gguf > $W/hf/combine.log 2>&1 < /dev/null || die "combine"
fi
( cd $W/hf && sha256sum imatrix.dat imatrix.gguf gemma-4-26B-A4B-agentic-F16.gguf omk-artifacts/26b_agentic_iq2_r16k/cache/eval.pt > SHA256_INPUTS ) || die sha
say "AB_INPUTS $(cut -c1-16 $W/hf/SHA256_INPUTS | paste -sd' ') (expect imatrix.dat 609a0a04, imatrix.gguf 80e56b9b)"
grep -q "^609a0a04.* imatrix.dat" $W/hf/SHA256_INPUTS && grep -q "^80e56b9b.* imatrix.gguf" $W/hf/SHA256_INPUTS || die "imatrix sha mismatch vs the card"
[ -s $RES ] || printf "tier\tarm\tsize_MiB\tklk\ttop1\n" > $RES
for T in $TIERS; do
  for A in $ARMS; do
    D=$W/$T; mkdir -p $D; G=$D/$A.gguf
    if [ ! -s $G ]; then
      say "AB_QUANT $T $A"
      $QUANT --imatrix ${IM[$A]} --tensor-type "blk\.\d+\.attn_(q|k|v|output)\.weight=$(attn_type $T)" $F16 $D/$A.partial.gguf $T 16 > $D/$A.quant.log 2>&1 < /dev/null \
        || { tail -5 $D/$A.quant.log >> $LOG; die "quant $T $A"; }
      mv $D/$A.partial.gguf $G
    fi
    if [ ! -s $D/$A.eval.json ]; then
      say "AB_SCORE $T $A"
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
        $PY $R26/step0_gguf_student_eval.py --gguf $G --eval $EVAL --bin $TOPK --omk $OMK --topk 256 --device cuda:0 --out $D/$A.eval.json \
        > $D/$A.eval.log 2>&1 < /dev/null && grep -q "GGUF_STUDENT_EVAL OK" $D/$A.eval.log || { tail -8 $D/$A.eval.log >> $LOG; die "score $T $A"; }
    fi
    r=$(python3 -c "import json; r=json.load(open('$D/$A.eval.json')); print('%.5f\t%.4f' % (r['klk'], r['top1']))")
    printf "%s\t%s\t%d\t%s\n" $T $A $(( $(stat -c %s $G) / 1048576 )) "$r" >> $RES
    say "AB_RESULT $T $A $r"
  done
done
say "AB_DONE arms [$ARMS]"; cat $RES | tee -a $LOG
