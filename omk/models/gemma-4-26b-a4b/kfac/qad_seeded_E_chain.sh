#!/bin/bash
# bm-892 seeded retrain (owner 2026-11-16 10:34 "prepare the seeded retrain, layout E, go when the probe is in";
# 10:57 "delete all three rows and go"). One chain on one GPU:
#   1. wait for the arm-E re-rounding that writes the seed (kfac_round --save-latents -> E_latents/layer_NN.pt, KFACR_DONE)
#   2. SEED GATE: export the UNTRAINED seeded student (--init-latents, no checkpoint) at layout E with the shipping
#      encoders, score it, and require (a) gate_up bytes identical to the K-FAC file E_gptq.gguf and (b) klk within
#      0.0005 of E_gptq's 0.0576 -- the run must start exactly where the rounding ended, or the seed is wrong
#   3. train: qad_C recipe (bs 1, accum 1, seq 4096, lr 2e-6 cosine -> 1e-7, warmup 50, --loss kl, anchor-as-teacher,
#      mtp 0.3), 600 steps, refresh 4 (probe: 89.2 s/step -> ~15 h), eval every 100; a plateau watcher drains the run
#      when a hundred steps gain < 0.001 klk. Snapshots OFF (64 GB each; disk), last.pt every 100 steps.
#   4. export last/final -> GGUF, score. Sentinels QSE_START / QSE_GATE_PASS|FAIL / QSE_TRAIN / QSE_PLATEAU / QSE_DONE / QSE_FAIL.
# Attention rule (owner 10:58): Gemma-4 A4B tiers below Q4_K_M carry ALL attention at Q6_K, Q4_K_M and above at Q8_0.
# IQ2_XXS is below -> the TRAINING layout is E with Q6_K attention (layout_E_attn_Q6_K). The seed gate still exports at
# E (Q8_0) because that is the layout the K-FAC reference was scored at (exact match), then exports the seed once more
# at the training layout: that file is the untrained "K-FAC + Q6_K attention" baseline the QAD has to beat.
# Never overwrites a run or a GGUF. Usage: qad_seeded_E_chain.sh <gpu>
set -uo pipefail
GPU=${1:?gpu index}
R=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_iq2_r16k; K=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac
PY=/srv/ml/envs/envs/omk-modelopt/bin/python; OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-ggmlenc
LC=/mnt/sdc/ml/llama.cpp-b12386; MODEL=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
SEED=$K/round/E_latents; REFGGUF=$K/round/E_gptq.gguf; REFJSON=$K/round/E_gptq.eval.json
LAY_GATE=$R/layout_E/layout.pt; LAY_TRAIN=$R/layout_E_attn_Q6_K/layout.pt
OUT=$R/qadE6_kfacseed; W=$R/seedgate_E; STEPS=${STEPS:-600}; REFRESH=${REFRESH:-4}
export PYTHONPATH=$OMK CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $W; LOG=$W/chain.driver.log
say() { echo "[qse $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "QSE_FAIL $*"; exit 1; }
COMMON="--mode qad --model $MODEL --cache $R/cache --device cuda:0 --llama-cpp $LC
  --ggml-encode IQ2_XXS,IQ2_S,Q8_0 --imatrix $R/cache/imatrix.pt --imatrix-convention msq --train-scope gemma4
  --export-only embedding --latent-dtype fp16 --mtp train --topk 64"
score() {  # <gguf> <json> <log>
  $PY $R/step0_gguf_student_eval.py --gguf $1 --eval $R/cache/eval.pt --bin /mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk \
    --omk $OMK --topk 256 --device cuda:0 --out $2 > $3 2>&1 < /dev/null || { tail -15 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f %.4f' % (r['klk'], r['top1']))"; }
[ -s $LAY_GATE ] && [ -s $LAY_TRAIN ] || die "no layout_E / layout_E_attn_Q6_K"
[ -s $REFGGUF ] && [ -s $REFJSON ] || die "no K-FAC reference $REFGGUF / $REFJSON"
[ -e $OUT ] && die "$OUT exists (never overwritten)"
REUSE_SEED=0; [ -s $W/seed_export.gguf ] && REUSE_SEED=1  # a finished seed export from an earlier attempt is reused, never rewritten
# the two layouts must differ ONLY in the attention projections' type (Q8_0 -> Q6_K)
python3 - "$(dirname $LAY_GATE)/layout.json" "$(dirname $LAY_TRAIN)/layout.json" <<'PY' | tee -a $LOG || die "layout pair check"
import sys, json
a, b = (json.load(open(p)) for p in sys.argv[1:3])
skip = {"seconds", "sizes_mib", "type_counts", "model_dir"}
flat = lambda d: {f"{k}.{kk}" if isinstance(v, dict) else k: (vv if isinstance(v, dict) else v)
                  for k, v in d.items() if k not in skip for kk, vv in (v.items() if isinstance(v, dict) else [(None, v)])}
fa, fb = flat(a), flat(b)
assert set(fa) == set(fb), f"layout keys differ: {sorted(set(fa) ^ set(fb))[:5]}"
diff = {k: (fa[k], fb[k]) for k in fa if fa[k] != fb[k]}
bad = {k: v for k, v in diff.items() if "attention" not in k.lower() and "attn" not in k.lower()}
assert not bad, f"non-attention differences: {list(bad.items())[:3]}"
assert diff, "the two layouts are identical"
for k, (x, y) in diff.items():
    assert "q8_0" in str(x).lower() and "q6_k" in str(y).lower(), f"{k}: {x} -> {y} is not Q8_0 -> Q6_K"
print(f"LAYOUT_PAIR_OK attention Q8_0 -> Q6_K ({', '.join(diff)}); the other {len(fa) - len(diff)} layout entries identical")
PY
say "QSE_START gpu $GPU steps $STEPS refresh $REFRESH seed $SEED ref $(klk_of $REFJSON); train layout $LAY_TRAIN"
# 1. the seed
for i in $(seq 1 480); do grep -q "KFACR_DONE\|KFACR_FAIL\|Error\|Traceback" $K/round/E_latents.log 2>/dev/null && break; sleep 15; done
grep -q "KFACR_DONE" $K/round/E_latents.log || die "rounding did not finish: $(tail -2 $K/round/E_latents.log | cut -c1-200)"
[ "$(ls $SEED/layer_*.pt | wc -l)" = 30 ] || die "seed has $(ls $SEED | wc -l) layer files, not 30"
say "seed ready: $(grep KFACR_DONE $K/round/E_latents.log | cut -c1-160)"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9); [ "${free:-0}" -ge 180 ] || die "only ${free}G free on /mnt/sdc (need 180: last.pt + its tmp + final)"
for i in $(seq 1 120); do used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] && break; sleep 30; done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU busy ($used MiB)"
# 2. the seed gate
if [ $REUSE_SEED = 1 ]; then
  say "reusing $W/seed_export.gguf from the earlier attempt ($(grep -c SEED_LATENTS $W/export.log) SEED line in its export.log)"
else
  $PY -m omk_qad.train --phase export $COMMON --layout $LAY_GATE --out $W/run_export --init-latents $SEED --gguf $W/seed_export.gguf \
    > $W/export.log 2>&1 < /dev/null || { tail -25 $W/export.log | tee -a $LOG; die "seed export"; }
fi
grep -q "SEED_LATENTS .*30 gate_up" $W/export.log || die "export did not seed: $(grep -c SEED_LATENTS $W/export.log) SEED lines"
grep -q "encoders: libggml.*Q8_0" $W/export.log || die "Q8_0 not through libggml"
say "seed export: $(grep SEED_LATENTS $W/export.log | cut -c1-200)"
$PY $R/../gguf_tensor_diff.py $W/seed_export.gguf $REFGGUF > $W/bytes_vs_kfac.txt 2>&1 || die "gguf_tensor_diff"
# Byte check. The K-FAC file was encoded from fp32 W_adj, the trainer holds fp16 latents: re-encoding the fp16 seed
# flips ~0.3 % of the IQ2_XXS bytes (measured 11:50: trainer export == fp16 re-encode to 0.0000, both 0.0030 from the
# fp32 file). So: attention must be byte-identical (115 tensors), routed IQ2_XXS may differ by <= 0.5 % of bytes;
# the score check below (|dklk| <= 0.0005) is the real gate.
python3 - $W/bytes_vs_kfac.txt <<'PY' | tee -a $LOG || { cut -c1-160 < $W/bytes_vs_kfac.txt | head -20 >> $LOG; die "seed bytes"; }
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"^attention\s+(\d+)\s+(\d+)\s+(\d+)", t, re.M); assert m and m.group(2) == "0" and m.group(3) == "0", f"attention not identical: {m.group(0) if m else 'no row'}"
m = re.search(r"bytes differing in routed_experts/IQ2_XXS: n=(\d+) mean frac ([\d.]+) min ([\d.]+) max ([\d.]+)", t)
mx = float(m.group(4)) if m else 0.0
assert mx <= 0.005, f"routed IQ2_XXS bytes differ by up to {mx:.4f} (> 0.005): not an fp16 effect"
print(f"SEED_BYTES_OK attention identical; routed IQ2_XXS max byte-diff frac {mx:.4f} (fp16 latent storage)")
PY
score $W/seed_export.gguf $W/seed_export.json $W/eval.log || die "seed score"
read -r sk st <<< "$(klk_of $W/seed_export.json)"; read -r rk rt <<< "$(klk_of $REFJSON)"
python3 -c "import sys; sys.exit(0 if abs($sk-$rk) <= 0.0005 else 1)" \
  || { say "QSE_GATE_FAIL seeded export $sk/$st vs K-FAC $rk/$rt (|dklk| > 0.0005)"; die "seed score mismatch"; }
say "QSE_GATE_PASS seeded untrained export $sk/$st vs K-FAC $rk/$rt (|dklk| <= 0.0005); attention bytes identical, routed IQ2_XXS within the fp16 margin"
# 2b. the same seed at the TRAINING layout (Q6_K attention): the untrained baseline the QAD must beat
$PY -m omk_qad.train --phase export $COMMON --layout $LAY_TRAIN --out $W/run_export_q6 --init-latents $SEED --gguf $W/seed_export_attnQ6K.gguf \
  > $W/export_q6.log 2>&1 < /dev/null || { tail -25 $W/export_q6.log | tee -a $LOG; die "seed export Q6_K"; }
# Q6_K is not a ModelOpt type: it is libggml-encoded by construction (--ggml-encode names only ModelOpt types)
grep -q "encoders: libggml.*Q8_0" $W/export_q6.log || die "Q8_0 not through libggml: $(grep encoders: $W/export_q6.log)"
sha256sum $W/seed_export_attnQ6K.gguf | tee $W/seed_export_attnQ6K.gguf.sha256 | tee -a $LOG
score $W/seed_export_attnQ6K.gguf $W/seed_export_attnQ6K.json $W/eval_q6.log || die "seed Q6_K score"
read -r bk bt <<< "$(klk_of $W/seed_export_attnQ6K.json)"
say "QSE_BASELINE K-FAC seed at Q6_K attention: $bk/$bt ($(stat -c %s $W/seed_export_attnQ6K.gguf | awk '{printf "%.2fGB", $1/1e9}'); Q8_0 attention $sk/$st; PTQ P6 0.0691)"
# 3. train (W&B: the bm-892 project; name says what it is)
say "QSE_TRAIN $OUT: $STEPS steps, refresh $REFRESH, eval/save every 100, snapshots off, layout E + Q6_K attention"
$PY -m omk_qad.train --phase train $COMMON --layout $LAY_TRAIN --out $OUT --init-latents $SEED \
  --seq 4096 --n-seq 16000 --bs 1 --accum 1 --steps $STEPS --lr 2e-6 --fp-lr 2e-6 --min-lr 1e-7 --warmup 50 \
  --beta1 0 --beta2 0.999 --clip 1.0 --loss kl --anchor-as-teacher --mtp-weight 0.3 \
  --refresh $REFRESH --ggml-refresh $REFRESH --eval-every 100 --save-every 100 --snapshot-every 0 --log-every 10 \
  --wandb-project omk-qad-26b-agentic-iq2-qad --wandb-name qadE6_kfacseed_r$REFRESH > $OUT.train.log 2>&1 < /dev/null &
TP=$!
sleep 600
kill -0 $TP 2>/dev/null || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer died in the first 10 min"; }
grep -q "SEED_LATENTS .*30 gate_up" $OUT.train.log || die "trainer did not seed"
say "trainer pid $TP alive; $(grep -m1 SEED_LATENTS $OUT.train.log | cut -c1-160)"
# plateau watcher: EVAL lines are JSON with klk; init then step100, step200 ...
drained=0
while kill -0 $TP 2>/dev/null; do
  sleep 300
  n=$(grep -c "^EVAL " $OUT.train.log)
  if [ "$n" -ge 3 ] && [ $drained = 0 ]; then
    gain=$(grep "^EVAL " $OUT.train.log | tail -2 | python3 -c "
import sys, json
r = [json.loads(l.split(' ', 1)[1]) for l in sys.stdin]
print('%.5f' % (r[0]['klk'] - r[1]['klk']))")
    if python3 -c "import sys; sys.exit(0 if $gain < 0.001 else 1)"; then
      say "QSE_PLATEAU last hundred steps gained $gain klk (< 0.001): draining at the next boundary"
      echo drain > $OUT/stage1/control; drained=1
    fi
  fi
done
wait $TP; rc=$?
grep "^EVAL " $OUT.train.log | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l.split(' ', 1)[1]); print('  eval', r.get('tag', '?'), 'klk %.4f top1 %.4f' % (r['klk'], r['top1']))" | tee -a $LOG
CK=$OUT/stage1/final.pt; [ -s $CK ] || CK=$OUT/stage1/last.pt
[ -s $CK ] || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer exit $rc, no checkpoint"; }
say "trainer exit $rc (drained=$drained); exporting $CK"
# 4. export + score
$PY -m omk_qad.train --phase export $COMMON --layout $LAY_TRAIN --out $OUT/export_run --load $CK --gguf $OUT/qadE6_kfacseed.gguf \
  > $OUT/export.log 2>&1 < /dev/null || { tail -25 $OUT/export.log | tee -a $LOG; die "final export"; }
sha256sum $OUT/qadE6_kfacseed.gguf | tee $OUT/qadE6_kfacseed.gguf.sha256 | tee -a $LOG
score $OUT/qadE6_kfacseed.gguf $OUT/qadE6_kfacseed.eval.json $OUT/eval.log || die "final score"
say "QSE_DONE seeded QAD (Q6_K attn) $(klk_of $OUT/qadE6_kfacseed.eval.json) | K-FAC seed at Q6_K attn $bk/$bt | K-FAC at Q8_0 $rk/$rt | PTQ P6 0.0691 E 0.0675 | kl7 (correct encoders) 0.0731 | $(stat -c %s $OUT/qadE6_kfacseed.gguf | awk '{printf "%.2fGB", $1/1e9}') | checkpoint $CK"
