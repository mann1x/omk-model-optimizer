#!/usr/bin/env bash
# bm-892/895: what K-FAC rounding of the DOWN experts + dense MLP adds on top of the gate_up K-FAC arm E.
#   E_gptq.gguf (gate_up K-FAC, rest llama-quantize PTQ, 0.0576/0.9469)
#     -> E_all_gptq.gguf: a copy with blk.*.ffn_down_exps / ffn_gate / ffn_up / ffn_down re-rounded by K-FAC GPTQ
#        at the SAME types (IQ4_NL / Q4_K / IQ4_NL): identical size, rounding-only difference.
# Rounding on the CPU (no GPU claim; GPU0 = 20B imatrix, GPU1 = seeded QAD); scoring on GPU0 (2 min, 10 GB).
# Sentinels: KDD_START / KDD_DONE / KDD_FAIL.  Never overwrites: refuses an existing output.
set -u
K=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac
R=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w
PY=/srv/ml/envs/envs/omk-modelopt/bin/python
LLAMA=/mnt/sdc/ml/llama.cpp-b12386
BASE=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal-BF16-eos106.gguf
REF=$K/round/E_gptq.gguf
OUT=$K/round/E_all_gptq.gguf
LOG=$K/round/E_all_chain.log
say() { echo "[kdd $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "KDD_FAIL $*"; exit 1; }
say "KDD_START ref $REF out $OUT (down experts + dense MLP, K-FAC GPTQ, CPU)"
[ -s $REF ] || die "no $REF"
[ -e $OUT ] && die "$OUT exists (never overwritten)"
avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9); [ "$avail" -ge 180 ] || die "/mnt/sdc ${avail}G < 180G"
cd $OMK || die cd
PYTHONPATH=$OMK OMP_NUM_THREADS=16 $PY -m omk_qad.kfac_round --factors $K/factors_w512_s2048_g10 --base $BASE \
  --imatrix $R/cache/imatrix.gguf --qtype auto --stems down_dense --ref $REF --out $OUT --device cpu \
  --llama-cpp $LLAMA --report $K/round/E_all_gptq.json > $K/round/E_all_gptq.log 2>&1 < /dev/null
grep -q "KFACR_DONE" $K/round/E_all_gptq.log || { tail -5 $K/round/E_all_gptq.log | tee -a $LOG; die "rounding"; }
grep "KFACR_STEM\|KFACR_DONE\|KFACR_ENCODER_MISMATCH" $K/round/E_all_gptq.log | tee -a $LOG
sha256sum $OUT | tee $OUT.sha256 | tee -a $LOG
# score on GPU0 next to the imatrix (10 GB, ~2 min)
export PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0
$PY $R/step0_gguf_student_eval.py --gguf $OUT --eval $R/cache/eval.pt --bin /mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk \
  --omk $OMK --topk 256 --device cuda:0 --out $K/round/E_all_gptq.eval.json > $K/round/E_all_gptq.eval.log 2>&1 < /dev/null \
  || { tail -15 $K/round/E_all_gptq.eval.log | tee -a $LOG; die "score"; }
grep -q "GGUF_STUDENT_EVAL OK" $K/round/E_all_gptq.eval.log || die "score sentinel"
python3 - $K/round/E_gptq.eval.json $K/round/E_all_gptq.eval.json <<'PYX' | tee -a $LOG
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(f"KDD_RESULT gate_up-only K-FAC {a['klk']:.4f}/{a['top1']:.4f}  + down&dense K-FAC {b['klk']:.4f}/{b['top1']:.4f}  "
      f"dklk {b['klk']-a['klk']:+.4f}  (PTQ twin E 0.0675/0.9396; same size)")
PYX
say "KDD_DONE $OUT"
