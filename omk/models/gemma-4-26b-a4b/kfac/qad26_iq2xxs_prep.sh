#!/bin/bash
# qad26_iq2xxs_prep.sh -- RUNS ON bs2. Prepares the 26B IQ2_XXS tier (the seeded QAD export, QS6_DONE 2026-11-16 21:09Z)
# for the K-FAC publisher: (1) the stock twin of the SAME layout E6 (llama-quantize b12386, cache imatrix, token
# embedding Q5_K, routed gate_up IQ2_XXS / down IQ4_NL, dense Q4_K / IQ4_NL, attention Q6_K, preset IQ2_M) on the CPU,
# (2) its score on GPU1 (free since the 26B trainer ended; the 20B chain's train stage waits for GPU1 < 2 GB and simply
# starts after), (3) ladder/IQ2_XXS/{qad.gguf -> the export, qad.gguf.sha256, qad.eval.json, ptq.eval.json, qad.done}.
# Never overwrites; the export itself is never touched. Sentinels QP_START / QP_STEP / QP_DONE / QP_FAIL in prep.log.
set -uo pipefail
R26=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
K=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac
PY=/srv/ml/envs/envs/omk-modelopt/bin/python
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
LLAMA=/mnt/sdc/ml/llama.cpp-b12386; QUANT=$LLAMA/build/bin/llama-quantize
TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
BASE=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal-BF16-eos106.gguf
IMAT=$R26/cache/imatrix.gguf; EVAL=$R26/cache/eval.pt
QAD=$R26/qadE6_kfacseed/qadE6_kfacseed_s600.gguf
D=$K/ladder/IQ2_XXS; mkdir -p $D; LOG=$D/prep.log
say() { echo "[qp $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "QP_FAIL $*"; exit 1; }
say "QP_START export $QAD"
for f in $QAD $QAD.sha256 ${QAD%.gguf}.eval.json $BASE $IMAT $EVAL $TOPK; do [ -s $f ] || die "missing $f"; done
grep -q "GGUF_STUDENT_EVAL OK" ${QAD%.gguf}.eval.log 2>/dev/null || say "note: no eval log beside the export (eval json present)"
P=$R26/step1/E6_ptq.gguf
if [ -s $P ]; then say "stock twin exists: $P"; else
  TT=(--tensor-type 'ffn_gate_up_exps\.weight=iq2_xxs' --tensor-type 'ffn_down_exps\.weight=iq4_nl' --tensor-type 'ffn_gate\.weight=q4_k'
      --tensor-type 'ffn_up\.weight=q4_k' --tensor-type 'ffn_down\.weight=iq4_nl' --tensor-type 'attn_q\.weight=q6_k'
      --tensor-type 'attn_k\.weight=q6_k' --tensor-type 'attn_v\.weight=q6_k' --tensor-type 'attn_output\.weight=q6_k')
  free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9); [ "$free" -ge 40 ] || die "only ${free}G free"
  say "QP_STEP stock twin E6 (llama-quantize IQ2_M + E6 tensor types, CPU 8 threads)"
  CUDA_VISIBLE_DEVICES="" $QUANT --imatrix $IMAT --token-embedding-type q5_K "${TT[@]}" $BASE $P.partial IQ2_M 8 > $R26/step1/E6_ptq.quant.log 2>&1 < /dev/null \
    || { tail -5 $R26/step1/E6_ptq.quant.log | tee -a $LOG; die "llama-quantize"; }
  mv $P.partial $P || die mv
  sha256sum $P > $P.sha256
fi
sq=$(stat -c %s $QAD); sp=$(stat -c %s $P); say "sizes: export $sq stock twin $sp"
[ "$sq" = "$sp" ] || say "WARNING sizes differ (export vs stock twin): layouts are not identical, check the types"
if [ -s $D/ptq.eval.json ]; then say "ptq.eval.json exists"; else
  n=0; while [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 60000 ]; do
    [ $((n % 20)) = 0 ] && say "GPU1 busy ($(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits) MiB), waiting"; n=$((n+1)); sleep 30; done
  say "QP_STEP score stock twin on GPU1"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    $PY $R26/step0_gguf_student_eval.py --gguf $P --eval $EVAL --bin $TOPK --omk $OMK --topk 256 --device cuda:0 \
    --out $D/ptq.eval.json > $D/ptq.eval.log 2>&1 < /dev/null || { tail -12 $D/ptq.eval.log | tee -a $LOG; die "score"; }
  grep -q "GGUF_STUDENT_EVAL OK" $D/ptq.eval.log || die "score: no OK sentinel"
fi
[ -e $D/qad.gguf ] || ln -s $QAD $D/qad.gguf
s=$(cut -d' ' -f1 $QAD.sha256); [ ${#s} = 64 ] || die "bad export sha"
echo "$s  qad.gguf" > $D/qad.gguf.sha256
cp -n ${QAD%.gguf}.eval.json $D/qad.eval.json; cp -n ${QAD%.gguf}.export.json $D/qad.export.json 2>/dev/null
cp -n $R26/step1/E6_ptq.quant.log $D/ptq.log 2>/dev/null
echo "$s  $QAD  $(date -u +%FT%TZ)" > $D/qad.done
say "QP_DONE $(python3 -c "
import json; p=json.load(open('$D/ptq.eval.json')); q=json.load(open('$D/qad.eval.json'))
print('stock E6 %.5f/%.4f  QAD s600 %.5f/%.4f  dKL %+.1f%%' % (p['klk'], p['top1'], q['klk'], q['top1'], 100*(q['klk']-p['klk'])/p['klk']))")"
