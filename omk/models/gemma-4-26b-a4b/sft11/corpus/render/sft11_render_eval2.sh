#!/bin/bash
# bm-889 sft11: kimi31b EVAL split v2. sft2's overlap gate failed on 5 scored positions, all from source conversation
# 5f4821bc (2 eval rows). Those 2 rows are left out of the eval split (train untouched); the first eval render is moved to
# rejected_kimi31b_eval_v1/ (kept). Re-render as sft2 did, re-run the overlap gate and the window check.
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery; B=$G/sft11_build; R=$B/render; PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
fail(){ echo "SFT11_RENDER_FAIL $*"; exit 1; }
cd $B || exit 1
mkdir rejected_kimi31b_eval_v1 || fail "rejected_kimi31b_eval_v1 exists"
mv $R/kimi31b.eval.roles.jsonl $R/kimi31b.eval.roles.stats.json rejected_kimi31b_eval_v1/ || fail mv
grep -v '"id": "kimi31b/5f4821bc-9070-43cf-ad85-dc94d5cfa210/' $G/sft6_build/phase3/build_v2/conv_c8/kimi31b.eval.conv.jsonl > kimi31b.eval.v2.conv.jsonl
echo "EVAL_ROWS $(wc -l < kimi31b.eval.v2.conv.jsonl) of $(wc -l < $G/sft6_build/phase3/build_v2/conv_c8/kimi31b.eval.conv.jsonl)"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$G/omk_sft2_code $PY -m omk_qad.rollouts --model /mnt/sdc/ml/models/gemma-4-26B-A4B-it \
  --in kimi31b.eval.v2.conv.jsonl --history-reasoning strip --seq 16384 --pack --out $R/kimi31b.eval.roles.jsonl > render_kimi31b_eval2.log 2>&1 < /dev/null \
  || fail "render: $(tail -2 render_kimi31b_eval2.log | tr '\n' ' ')"
grep -q '"complete": true' $R/kimi31b.eval.roles.stats.json || fail incomplete
$PY sft2_overlap.py $R > overlap2.log 2>&1; cat overlap2.log
[ -e $R/OVERLAP_OK ] || fail "overlap gate"
python3 $G/gate_analysis_loop/window_check.py $R | tee window_check.log
grep -q "'nobos': 0," window_check.log || fail "a window starts mid-conversation"
grep -q "'cut_end': 0}" window_check.log || fail "a window ends mid-conversation"
N=$(cat $R/*.train.roles.stats.json | grep -o '"windows": [0-9]*' | awk '{s+=$2} END {print s}')
echo "SFT11_RENDER_OK $(date -u +%FT%TZ) train windows $N steps $((N / 2))"
