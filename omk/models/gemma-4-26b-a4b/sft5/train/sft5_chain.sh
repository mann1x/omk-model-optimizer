#!/bin/bash
# bm-889 owner 2026-11-02 16:44 CET "when the render is done, stop sft4 and launch sft5 with it"; 16:45 "delete all of sft4"
# (every sft4 checkpoint incl. step 200, and the sft4-s200 HF export weights; the step 200 GGUF, logs, json, W&B stay).
# 1 wait render  2 sft5_gates.py  3 overlap gate  4 one SIGTERM to sft4, PAUSED, GPU0 free  5 delete sft4 ckpts + export
# safetensors (not open)  6 launch sft5_train.sh  7 wait SFT5_START + step 1. Sentinels SFT5CHAIN_OK / SFT5CHAIN_FAIL <step>.
set -uo pipefail
cd /mnt/sdc/omk-model-optimizer/g4_recovery || exit 1
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python; R=sft4_build/render_32k_v3_mlast2; T=2060263
HF=/mnt/sdc/ml/models/gemma-4-26B-A4B-it-bm889-sft4-s200
say(){ echo "SFT5CHAIN $(date -u +%FT%TZ) $*"; }
fail(){ echo "SFT5CHAIN_FAIL $(date -u +%FT%TZ) $*"; exit 1; }
say "1 wait render"
until grep -q RENDER_ALL_DONE sft4_build/render_v3_mlast2.out; do sleep 30; done
grep -q FAIL sft4_build/render_v3_mlast2.out && fail "1 render: $(grep FAIL sft4_build/render_v3_mlast2.out | head -3 | tr '\n' ' ')"
say "2 gates"; (cd sft4_build && CUDA_VISIBLE_DEVICES="" $PY sft5_gates.py) || fail "2 gates"
say "3 overlap"; CUDA_VISIBLE_DEVICES="" $PY sft2_build/sft2_overlap.py $R || fail "3 overlap gate"
[ -e $R/OVERLAP_OK ] || fail "3 no OVERLAP_OK"
say "4 stop sft4 (pid $T)"
kill -0 $T 2>/dev/null || fail "4 sft4 trainer not running"
kill -TERM $T
for _ in $(seq 1 120); do kill -0 $T 2>/dev/null || break; sleep 10; done
kill -0 $T 2>/dev/null && fail "4 sft4 alive 20 min after SIGTERM (no second SIGTERM sent)"
grep -E "PAUSED at step" train_sft4.log | tail -1 | cut -c1-200
for _ in $(seq 1 24); do m=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 2000 ] && break; sleep 5; done
[ "$m" -lt 2000 ] || fail "4 GPU0 at $m MiB"
say "5 delete sft4 checkpoints and the s200 export weights"
F=$(ls runs/g4_bf16_sft4/stage1/*.pt runs/g4_bf16_sft4/early/*.pt $HF/*.safetensors 2>/dev/null)
for f in $F; do fuser -s "$f" 2>/dev/null && fail "5 $f is open"; done
ls -la $F | awk '{print $5, $9}'
rm -f $F; df -h /mnt/sdc | tail -1
say "6 launch sft5"
setsid nohup bash sft5_train.sh > sft5_train.out 2>&1 < /dev/null &
for _ in $(seq 1 120); do grep -qE "^step 1 |SFT5_FAIL" train_sft5.log sft5_train.out 2>/dev/null && break; sleep 30; done
grep -E "SFT5_(START|FAIL)" train_sft5.log sft5_train.out 2>/dev/null | tail -2
grep -q SFT5_FAIL train_sft5.log sft5_train.out 2>/dev/null && fail "6 $(grep -h SFT5_FAIL train_sft5.log sft5_train.out | tail -1)"
grep -E "^step 1 " train_sft5.log | cut -c1-200
grep -q "^step 1 " train_sft5.log || fail "6 no step 1 within 60 min"
echo "SFT5CHAIN_OK $(date -u +%FT%TZ)"
