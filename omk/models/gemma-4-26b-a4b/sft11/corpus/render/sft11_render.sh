#!/bin/bash
# bm-889 sft11 corpus (owner 2026-11-11 12:17 CET: "go ahead, bf16 like sft2, full epoch training, gate at 400"):
# sft2's render_v3 verbatim (copied from the solidpc archive, sha checked) + kimi31b (sft6 conv_c8, 693 train / 32 eval)
# rendered EXACTLY as sft2 rendered its corpus (sft2_chain3.sh:30): omk_sft2_code rollouts --history-reasoning strip
# --seq 16384 --pack. Then sft2's train/eval 200-gram overlap gate (writes OVERLAP_OK) and the bos/split window check.
# Sentinels SFT11_RENDER_OK / SFT11_RENDER_FAIL.
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery; R=$G/sft11_build/render; PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
C=$G/sft6_build/phase3/build_v2/conv_c8
fail(){ echo "SFT11_RENDER_FAIL $*"; exit 1; }
cd $G/sft11_build || exit 1
for s in train eval; do
  out=$R/kimi31b.$s.roles.jsonl
  [ -e $out ] && fail "$out exists"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$G/omk_sft2_code $PY -m omk_qad.rollouts --model /mnt/sdc/ml/models/gemma-4-26B-A4B-it \
    --in $C/kimi31b.$s.conv.jsonl --history-reasoning strip --seq 16384 --pack --out $out > render_kimi31b_$s.log 2>&1 < /dev/null \
    || fail "render kimi31b $s: $(tail -2 render_kimi31b_$s.log | tr '\n' ' ')"
  grep -q '"complete": true' ${out%.jsonl}.stats.json || fail "kimi31b $s render incomplete"
  echo "RENDERED kimi31b.$s $(grep -o '"rows": [0-9]*\|"windows": [0-9]*\|"refused_not_prefix": [0-9]*\|"context_only_turns": [0-9]*' ${out%.jsonl}.stats.json | tr '\n' ' ')"
done
$PY sft2_overlap.py $R > overlap.log 2>&1; tail -3 overlap.log
[ -e $R/OVERLAP_OK ] || fail "overlap gate: $(tail -2 overlap.log | tr '\n' ' ')"
python3 $G/gate_analysis_loop/window_check.py $R | tee window_check.log
grep -q "'nobos': 0," window_check.log || fail "a window starts mid-conversation"
grep -q "'cut_end': 0}" window_check.log || fail "a window ends mid-conversation"
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
echo "SFT11_RENDER_OK $(date -u +%FT%TZ) train windows $N steps $((N / 2))"
