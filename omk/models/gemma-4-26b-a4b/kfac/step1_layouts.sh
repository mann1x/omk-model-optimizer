#!/bin/bash
# bm-892 v2 step 1 (owner 2026-11-14 20:46 "go with the iq2 plan on gpu0"): PTQ layout sensitivity, no training.
# Each arm = llama-quantize (b12386) of the teacher's BF16 GGUF with the QAD's imatrix (cache/imatrix.gguf) and
# explicit per-class types, then step0_gguf_student_eval.py (the trainer's eval set and metrics, GGUF side) on GPU0.
#   A  current QAD layout: routed gate_up IQ2_XXS, routed down IQ4_NL, dense MLP gate/up IQ2_XXS + down IQ4_NL,
#      attn q/k IQ2_S, attn_output IQ3_S, attn_v Q4_K, token_embd Q5_K  (PTQ baseline; QAD step-0 klk was 0.4206)
#   B  A + dense MLP gate/up Q4_K              (the always-on path every token uses)
#   C  A + attention q/k/v/output Q8_0          (NVIDIA Bolt's map keeps attention at Q8_0)
#   D  A + routed gate_up IQ2_S                 (the bulk of the weights, 2.06 -> 2.5 bpw)
#   E  B + C
# Never overwrites (refuses an existing arm GGUF). Sentinels STEP1_ARM <arm> <size> <klk...> / STEP1_DONE / STEP1_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k; W=$R/step1; mkdir -p $W
LC=/mnt/sdc/ml/llama.cpp-b12386/build/bin; PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
SRC=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal-BF16-eos106.gguf; IM=$R/cache/imatrix.gguf
OMK=/mnt/sdc/omk-model-optimizer/code/omk-qad-g4r16k; BIN=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
EV=$R/cache/eval.pt
export CUDA_VISIBLE_DEVICES=0
base() {  # class=type pairs of layout A
  echo "ffn_gate_up_exps\.weight=iq2_xxs ffn_down_exps\.weight=iq4_nl ffn_gate\.weight=$1 ffn_up\.weight=$1 ffn_down\.weight=iq4_nl" \
       "attn_q\.weight=$2 attn_k\.weight=$2 attn_v\.weight=$3 attn_output\.weight=$4"
}
declare -A ARMS
ARMS[A]="$(base iq2_xxs iq2_s q4_k iq3_s)"
ARMS[B]="$(base q4_k iq2_s q4_k iq3_s)"
ARMS[C]="$(base iq2_xxs q8_0 q8_0 q8_0)"
ARMS[D]="$(base iq2_xxs iq2_s q4_k iq3_s | sed 's/ffn_gate_up_exps\\.weight=iq2_xxs/ffn_gate_up_exps\\.weight=iq2_s/')"
ARMS[E]="$(base q4_k q8_0 q8_0 q8_0)"
# step 1b (owner 21:36: "an iq2 that is smaller and better"): E with the routed down projections (4.28 GB at IQ4_NL,
# 7.6 B params) lower: F IQ3_S (~9.2 GB), G IQ3_XXS (~8.9), H IQ2_S (~8.4), I IQ2_XXS (~7.9, NVIDIA Bolt's choice)
for x in F:iq3_s G:iq3_xxs H:iq2_s I:iq2_xxs; do
  ARMS[${x%%:*}]="$(base q4_k q8_0 q8_0 q8_0 | sed "s/ffn_down_exps\\\\.weight=iq4_nl/ffn_down_exps\\\\.weight=${x#*:}/")"
done
echo "STEP1_START $(date -u +%FT%TZ)"
for arm in ${1:-A B C D E}; do
  out=$W/$arm.gguf
  [ -e $out ] && { echo "STEP1_FAIL $out exists"; exit 1; }
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  [ "$free" -ge 60 ] || { echo "STEP1_FAIL only ${free}G free"; exit 1; }
  tt=(); for p in ${ARMS[$arm]}; do tt+=(--tensor-type "$p"); done
  echo "[$arm] ${ARMS[$arm]}" | tee $W/$arm.layout
  CUDA_VISIBLE_DEVICES="" $LC/llama-quantize --imatrix $IM --token-embedding-type q5_K "${tt[@]}" $SRC $out IQ2_M 28 \
    > $W/$arm.quant.log 2>&1 < /dev/null || { tail -15 $W/$arm.quant.log; echo "STEP1_FAIL quantize $arm"; exit 1; }
  grep "applying manual override" $W/$arm.quant.log | awk '{print $NF}' | sort | uniq -c | tr '\n' ' ' > $W/$arm.types
  $PY $R/step0_gguf_student_eval.py --gguf $out --eval $EV --bin $BIN --omk $OMK --topk 256 --out $W/$arm.json \
    > $W/$arm.eval.log 2>&1 < /dev/null || { tail -15 $W/$arm.eval.log; echo "STEP1_FAIL eval $arm"; exit 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $W/$arm.eval.log || { echo "STEP1_FAIL eval $arm scored mismatch"; exit 1; }
  echo "STEP1_ARM $arm $(stat -c %s $out | awk '{printf "%.2fGB", $1/1e9}') $(python3 -c "
import json; r=json.load(open('$W/$arm.json')); f=r['families']
print('klk %.4f top1 %.4f agentic %.4f opencode %.4f hermes %.4f kimi31b %.4f' % (r['klk'], r['top1'], f['agentic.eval.roles']['klk'], f['opencode.eval.roles']['klk'], f['hermes.eval.roles']['klk'], f['kimi31b.eval.roles']['klk']))") types: $(cat $W/$arm.types)"
done
echo "STEP1_DONE $(date -u +%FT%TZ)"
