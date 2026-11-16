#!/bin/bash
# qad20_seeded_E_chain.sh — RUNS ON bs2. The 20B v9-agentic (sft12) IQ2_XXS QAD, seeded from a K-FAC rounding, the exact
# recipe of the 26B qadE6_kfacseed run (qad_seeded_E_chain.sh + _resume600.sh), owner 2026-11-16 20:43 CET "20B IQ2_XXS
# QAD is a go", 20:44 "how do you start the 20b qad without the iq2_xxs kfac?" -> the chain builds that seed first:
#   0. wait for the 20B bf16 HF weights (pulled from the pod, SHA256SUMS-verified)
#   1. CACHE (GPU0, ~3 h): teacher top-64 over the SAME corpus mix as the 26B cache (sft11 render roles, 4217 x 16384),
#      the 44-row eval set, and the corpus imatrix (imatrix.pt + imatrix.gguf) the PTQ ref, the rounder and the trainer
#      all share (a seed is only exact if every stage uses one imatrix)
#   2. LAYOUT E + Q6_K attention (omk_qad.layout): routed gate_up IQ2_XXS, routed down IQ4_NL, dense gate/up Q4_K,
#      dense down IQ4_NL, all attention Q6_K (owner's attention rule: below Q4_K_M -> Q6_K), rest IQ2_M
#   3. PTQ ref at that layout (llama-quantize b12386, CPU)              -> step1/E6.gguf
#   4. SEED: kfac_round gate_up IQ2_XXS gptq --save-latents (GPU0, ~40 min) -> round/E6_gptq.gguf + round/E6_latents, scored
#   5. SEED GATE: export the untrained seeded student, attention bytes identical, routed IQ2_XXS within the fp16 margin,
#      |dklk| <= 0.0005 vs the K-FAC file -- the run must start exactly where the rounding ended
#   6. TRAIN on GPU1 after the 26B resume-600 chain (QS6_DONE/QS6_FAIL) and its GPU is free: 600 steps, lr 2e-6 cosine
#      -> 1e-7, warmup 50, kl, anchor-as-teacher, refresh 4, eval/save every 100, snapshots off, NO plateau rule
#      (owner 2026-11-16 18:47 on the 26B: run to 600 and look). W&B project omk-qad-20b-v9agentic-iq2-qad.
#   7. export final.pt (or last.pt) -> qadE6_kfacseed_s600.gguf, score.
# Disk (/mnt/sdc): cache >= 40 G, seed >= 50 G, train >= 150 G (last.pt ~47 G + its tmp + final); the chain WAITS and
# logs QAD20_DISK_WAIT -- it never deletes anything. Never overwrites a run or a GGUF.
# Sentinels QAD20_START / QAD20_STEP / QAD20_DISK_WAIT / QAD20_GATE_PASS|FAIL / QAD20_TRAIN / QAD20_DONE / QAD20_FAIL.
set -uo pipefail
RUNS=/mnt/sdc/omk-model-optimizer/runs
R=$RUNS/g4_20b_iq2; K=$RUNS/g4_20b_kfac; R26=$RUNS/g4_agentic_iq2_r16k
PY=/srv/ml/envs/envs/omk-modelopt/bin/python
OMK=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2cuda      # trainer / cache / layout / export: the tree the 26B run uses
OMKR=/mnt/sdc/omk-model-optimizer/code/omk-wip-iq2w        # kfac_round with --stems / --save-latents (the ladder tree)
LC=/mnt/sdc/ml/llama.cpp-b12386; QUANT=$LC/build/bin/llama-quantize
TOPK=/mnt/sdc/omk-model-optimizer/llama_topk_b12386/llama_topk
MODEL=/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12
BASE=/srv/ml/scratch/sft12_ac_imatrix/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12-F16.gguf
FACTORS=$K/factors_w512_s2048_g10
REND=/mnt/sdc/omk-model-optimizer/g4_recovery/sft11_build/render
PREV=$R26/qad_seeded_E_chain_resume600.launcher.out        # the 26B run on GPU1: QS6_DONE / QS6_FAIL frees the GPU
LAY=$R/layout_E_attn_Q6_K/layout.pt
OUT=$R/qadE6_kfacseed; W=$R/seedgate_E6; STEPS=${STEPS:-600}; REFRESH=${REFRESH:-4}
mkdir -p $R $W $R/step1 $R/round; LOG=$R/chain.driver.log
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
say() { echo "[qad20 $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "QAD20_FAIL $*"; exit 1; }
disk() { df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9; }
need_disk() { local n=0; while [ "$(disk)" -lt $1 ]; do [ $((n % 20)) = 0 ] && say "QAD20_DISK_WAIT /mnt/sdc $(disk)G < $1G before $2 (nothing is deleted by this chain)"; n=$((n + 1)); sleep 30; done; }
gpu_used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits; }
klk_of() { python3 -c "import json; r=json.load(open('$1')); print('%.4f %.4f' % (r['klk'], r['top1']))"; }
score() {  # <gguf> <json> <log> <gpu>
  CUDA_VISIBLE_DEVICES=$4 PYTHONPATH=$OMK $PY $R26/step0_gguf_student_eval.py --gguf $1 --eval $R/cache/eval.pt --bin $TOPK \
    --omk $OMK --topk 256 --device cuda:0 --out $2 > $3 2>&1 < /dev/null || { tail -15 $3 | tee -a $LOG; return 1; }
  grep -q "GGUF_STUDENT_EVAL OK" $3
}
DATA="$REND/agentic.train.roles.jsonl=14.87,$REND/opencode.train.roles.jsonl=22.90,$REND/ifchat_v1.train.roles.jsonl=5.73,$REND/hermes.train.roles.jsonl=3.29,$REND/ifchat_v2.train.roles.jsonl=2.84,$REND/kimi31b.train.roles.jsonl=2.64,$REND/swe.train.roles.jsonl=1.03,$REND/multilingual.train.roles.jsonl=0.62,$REND/math.train.roles.jsonl=0.29,$REND/science.train.roles.jsonl=0.26,$REND/cp.train.roles.jsonl=0.19"
EVALD=""; for f in agentic opencode ifchat_v1 hermes ifchat_v2 kimi31b swe multilingual math science cp; do EVALD="$EVALD${EVALD:+,}$REND/$f.eval.roles.jsonl"; done
COMMON="--mode qad --model $MODEL --cache $R/cache --device cuda:0 --llama-cpp $LC
  --ggml-encode Q8_0 --imatrix $R/cache/imatrix.pt --imatrix-convention msq --train-scope gemma4
  --export-only embedding --latent-dtype fp16 --mtp train --topk 64"
[ -e $OUT ] && die "$OUT exists (never overwritten)"
[ -s $BASE ] || die "no F16 base $BASE"; ls $FACTORS/layer_29.pt > /dev/null 2>&1 || die "20B factors incomplete"
for f in $(tr ',' '\n' <<<"$DATA" | cut -d= -f1) $(tr ',' '\n' <<<"$EVALD"); do [ -s $f ] || die "corpus file missing: $f"; done
say "QAD20_START steps $STEPS refresh $REFRESH model $MODEL base $BASE factors $FACTORS"
# 0. weights
for i in $(seq 1 240); do ps -eo args | grep -q "[r]sync .*v9agentic/bf16/ $MODEL/" || break; [ "$i" = 240 ] && die "bf16 pull still running after 2 h"; sleep 30; done
[ -s $MODEL/config.json ] && [ -s $MODEL/SHA256SUMS ] || die "pulled weights incomplete at $MODEL ($(ls $MODEL 2>/dev/null | wc -l) files)"
(cd $MODEL && sha256sum -c SHA256SUMS --ignore-missing --quiet) || die "pulled weights fail SHA256SUMS"
say "QAD20_STEP weights OK ($(du -sh $MODEL | cut -f1), $(ls $MODEL/*.safetensors | wc -l) shards)"
# 1. cache (GPU0)
if [ "$(python3 -c "import json;print(json.load(open('$R/cache/state.json')).get('status',''))" 2>/dev/null)" = done ] && [ -s $R/cache/imatrix.pt ] && [ -s $R/cache/eval.pt ]; then
  say "cache exists, kept"
else
  need_disk 100 cache   # the 26B run saves a 61 G last.pt until it ends; the cache adds ~22 G
  say "QAD20_STEP cache on GPU0 (4217 x 16384, topk 64, 11 sources; ~3 h)"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$OMK $PY -m omk_qad.train --phase cache --mode qad --model $MODEL --cache $R/cache \
    --data "$DATA" --eval-data "$EVALD" --seq 16384 --n-seq 4217 --topk 64 --shard 250 \
    --train-scope gemma4 --latent-dtype fp16 --export-only embedding --mtp train --device cuda:0 --llama-cpp $LC \
    > $R/cache.driver.log 2>&1 < /dev/null || { tail -20 $R/cache.driver.log | tee -a $LOG; die "cache phase"; }
  [ "$(python3 -c "import json;print(json.load(open('$R/cache/state.json')).get('status',''))")" = done ] || die "cache state not done"
  [ -s $R/cache/imatrix.pt ] && [ -s $R/cache/imatrix.gguf ] && [ -s $R/cache/eval.pt ] || die "cache incomplete (imatrix.pt/gguf, eval.pt)"
  say "cache: $(grep -E '^eval:|IMATRIX from' $R/cache/cache.log | cut -c1-160 | tr '\n' ' ')"
fi
# 2. layout
if [ -s $LAY ]; then say "layout exists, kept"; else
  say "QAD20_STEP layout E + Q6_K attention"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OMK $PY -m omk_qad.layout --model $MODEL --out $LAY --llama-cpp $LC --rest IQ2_M \
    --fixed routed_experts=iq2_xxs --no-mtp \
    --override-regex 'ffn_down_exps\.weight$=IQ4_NL' --override-regex 'ffn_gate\.weight$=Q4_K' --override-regex 'ffn_up\.weight$=Q4_K' \
    --override-regex 'ffn_down\.weight$=IQ4_NL' --override-regex 'attn_q\.weight$=Q6_K' --override-regex 'attn_k\.weight$=Q6_K' \
    --override-regex 'attn_v\.weight$=Q6_K' --override-regex 'attn_output\.weight$=Q6_K' > $R/layout_E_attn_Q6_K/build.driver.log 2>&1 < /dev/null \
    || { tail -20 $R/layout_E_attn_Q6_K/build.driver.log | tee -a $LOG; die "layout"; }
fi
python3 - $R/layout_E_attn_Q6_K/layout.json <<'PY' | tee -a $LOG || die "layout check"
import json, sys
d = json.load(open(sys.argv[1])); tc = d["type_counts"]; pc = d.get("per_class", {})
att = {t for c, v in pc.items() if "attn" in c.lower() or "attention" in c.lower() for t in v}
assert tc.get("IQ2_XXS") == 30 and tc.get("IQ4_NL") == 60 and tc.get("Q4_K") == 60, f"MLP types off: {tc}"
assert att == {"Q6_K"}, f"attention classes carry {att}, not only Q6_K"
print(f"LAYOUT_OK {tc} attention={sorted(att)} mtp={d.get('mtp')}")
PY
# 3. PTQ ref (CPU) at the same layout
E=$R/step1/E6.gguf
if [ -s $E ]; then say "PTQ ref exists, kept"; else
  need_disk 30 "PTQ ref"
  TT=(--tensor-type 'ffn_gate_up_exps\.weight=iq2_xxs' --tensor-type 'ffn_down_exps\.weight=iq4_nl' --tensor-type 'ffn_gate\.weight=q4_k'
      --tensor-type 'ffn_up\.weight=q4_k' --tensor-type 'ffn_down\.weight=iq4_nl' --tensor-type 'attn_q\.weight=q6_k'
      --tensor-type 'attn_k\.weight=q6_k' --tensor-type 'attn_v\.weight=q6_k' --tensor-type 'attn_output\.weight=q6_k')
  CUDA_VISIBLE_DEVICES="" $QUANT --imatrix $R/cache/imatrix.gguf --token-embedding-type q5_K "${TT[@]}" $BASE $E.partial IQ2_M 8 \
    > $R/step1/E6.quant.log 2>&1 < /dev/null || { tail -5 $R/step1/E6.quant.log | tee -a $LOG; die "PTQ ref"; }
  mv $E.partial $E && sha256sum $E > $E.sha256
  say "PTQ ref $(du -h $E | cut -f1); types: $(grep -oE 'type +[a-z0-9_]+' $R/step1/E6.quant.log | sort | uniq -c | sort -rn | head -6 | tr -s ' \n' ' ')"
fi
# 4. the seed (GPU0): gate_up IQ2_XXS K-FAC GPTQ, latents saved -- the 26B recipe (round/E_latents)
SEED=$R/round/E6_latents; REFG=$R/round/E6_gptq.gguf; REFJ=$R/round/E6_gptq.eval.json
if [ -s $REFG ] && [ "$(ls $SEED/layer_*.pt 2>/dev/null | wc -l)" = 30 ]; then say "seed exists, kept"; else
  need_disk 60 seed
  for i in $(seq 1 240); do [ "$(gpu_used 0)" -lt 60000 ] && break; sleep 30; done
  say "QAD20_STEP seed rounding on GPU0 (gate_up IQ2_XXS gptq, ref $E; ~40 min)"
  CUDA_VISIBLE_DEVICES=0 OMK_GGML_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=$OMKR $PY -m omk_qad.kfac_round --factors $FACTORS --base $BASE \
    --imatrix $R/cache/imatrix.gguf --qtype IQ2_XXS --method gptq --ref $E --out $REFG --report $R/round/E6_gptq.json \
    --save-latents $SEED --device cuda:0 --llama-cpp $LC > $R/round/E6_gptq.log 2>&1 < /dev/null \
    || { tail -8 $R/round/E6_gptq.log | tee -a $LOG; die "seed rounding"; }
  grep -q KFACR_DONE $R/round/E6_gptq.log || die "no KFACR_DONE: $(tail -2 $R/round/E6_gptq.log | cut -c1-200)"
  [ "$(ls $SEED/layer_*.pt | wc -l)" = 30 ] || die "seed has $(ls $SEED | wc -l) layer files, not 30"
  sha256sum $REFG > $REFG.sha256
  say "seed: $(grep KFACR_DONE $R/round/E6_gptq.log | cut -c1-160)"
fi
[ -s $REFJ ] || { score $REFG $REFJ $R/round/E6_gptq.eval.log 0 || die "seed score"; }
[ -s $R/step1/E6.eval.json ] || { score $E $R/step1/E6.eval.json $R/step1/E6.eval.log 0 || die "PTQ ref score"; }
say "QAD20_STEP K-FAC seed $(klk_of $REFJ) vs llama.cpp PTQ at the same layout $(klk_of $R/step1/E6.eval.json)"
# 5. seed gate
if [ -s $W/seed_export.gguf ]; then say "reusing $W/seed_export.gguf"; else
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$OMK $PY -m omk_qad.train --phase export $COMMON --layout $LAY --out $W/run_export --init-latents $SEED \
    --gguf $W/seed_export.gguf > $W/export.log 2>&1 < /dev/null || { tail -25 $W/export.log | tee -a $LOG; die "seed export"; }
fi
grep -q "SEED_LATENTS .*30 gate_up" $W/export.log || die "export did not seed: $(grep -c SEED_LATENTS $W/export.log) SEED lines"
grep -q "encoders: libggml.*Q8_0" $W/export.log || die "Q8_0 not through libggml: $(grep encoders: $W/export.log | cut -c1-160)"
PYTHONPATH=$OMK $PY $RUNS/gguf_tensor_diff.py $W/seed_export.gguf $REFG > $W/bytes_vs_kfac.txt 2>&1 || die "gguf_tensor_diff"
python3 - $W/bytes_vs_kfac.txt <<'PY' | tee -a $LOG || { cut -c1-160 < $W/bytes_vs_kfac.txt | head -20 >> $LOG; die "seed bytes"; }
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"^attention\s+(\d+)\s+(\d+)\s+(\d+)", t, re.M); assert m and m.group(2) == "0" and m.group(3) == "0", f"attention not identical: {m.group(0) if m else 'no row'}"
m = re.search(r"bytes differing in routed_experts/IQ2_XXS: n=(\d+) mean frac ([\d.]+) min ([\d.]+) max ([\d.]+)", t)
mx = float(m.group(4)) if m else 0.0
assert mx <= 0.005, f"routed IQ2_XXS bytes differ by up to {mx:.4f} (> 0.005): not an fp16 effect"
print(f"SEED_BYTES_OK attention identical; routed IQ2_XXS max byte-diff frac {mx:.4f} (fp16 latent storage)")
PY
[ -s $W/seed_export.json ] || { score $W/seed_export.gguf $W/seed_export.json $W/eval.log 0 || die "seed export score"; }
read -r sk st <<< "$(klk_of $W/seed_export.json)"; read -r rk rt <<< "$(klk_of $REFJ)"
python3 -c "import sys; sys.exit(0 if abs($sk-$rk) <= 0.0005 else 1)" \
  || { say "QAD20_GATE_FAIL seeded export $sk/$st vs K-FAC $rk/$rt (|dklk| > 0.0005)"; die "seed score mismatch"; }
say "QAD20_GATE_PASS seeded untrained export $sk/$st vs K-FAC $rk/$rt; attention bytes identical, routed IQ2_XXS within the fp16 margin. This is the untrained baseline the QAD must beat."
# 6. train on GPU1 after the 26B chain
for i in $(seq 1 2880); do grep -qE "QS6_DONE|QS6_FAIL" $PREV 2>/dev/null && break; [ "$i" = 2880 ] && die "26B chain not finished after 24 h"; sleep 30; done
say "26B chain: $(grep -E 'QS6_DONE|QS6_FAIL' $PREV | tail -1 | cut -c1-200)"
for i in $(seq 1 240); do ps -eo args | grep -q "[o]mk_qad.train --phase train.*g4_agentic_iq2_r16k" || break; sleep 30; done
for i in $(seq 1 240); do [ "$(gpu_used 1)" -lt 2000 ] && break; [ "$i" = 240 ] && die "GPU1 still busy ($(gpu_used 1) MiB)"; sleep 30; done
need_disk 150 train
say "QAD20_TRAIN $OUT: $STEPS steps, refresh $REFRESH, eval/save every 100, snapshots off, layout E + Q6_K attention, GPU1"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$OMK $PY -m omk_qad.train --phase train $COMMON --layout $LAY --out $OUT --init-latents $SEED \
  --seq 4096 --n-seq 16000 --bs 1 --accum 1 --steps $STEPS --lr 2e-6 --fp-lr 2e-6 --min-lr 1e-7 --warmup 50 \
  --beta1 0 --beta2 0.999 --clip 1.0 --loss kl --anchor-as-teacher --mtp-weight 0.3 \
  --refresh $REFRESH --ggml-refresh $REFRESH --eval-every 100 --save-every 100 --snapshot-every 0 --log-every 10 \
  --wandb-project omk-qad-20b-v9agentic-iq2-qad --wandb-name qadE6_kfacseed_r$REFRESH > $OUT.train.log 2>&1 < /dev/null &
TP=$!
sleep 600; kill -0 $TP 2>/dev/null || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer died in the first 10 min"; }
say "trainer pid $TP alive; $(grep -E '^step [0-9]+ ' $OUT.train.log | tail -1 | cut -c1-160)"
wait $TP; rc=$?
say "trainer exit $rc; evals: $(grep -c '^EVAL ' $OUT.train.log)"; grep '^EVAL ' $OUT.train.log | cut -c1-200 | tee -a $LOG
[ $rc = 0 ] || { tail -30 $OUT.train.log | tee -a $LOG; die "trainer rc $rc"; }
# 7. export + score
CK=$OUT/stage1/final.pt; [ -s $CK ] || CK=$OUT/stage1/last.pt; [ -s $CK ] || die "no final.pt / last.pt"
G=$R/qadE6_kfacseed_s$STEPS.gguf; [ -e $G ] && die "$G exists (never overwritten)"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$OMK $PY -m omk_qad.train --phase export $COMMON --layout $LAY --load $CK --out $R/export_s$STEPS --gguf $G \
  > $R/export_s$STEPS.log 2>&1 < /dev/null || { tail -20 $R/export_s$STEPS.log | tee -a $LOG; die "export"; }
sha256sum $G | tee $G.sha256 | tee -a $LOG
score $G $R/qadE6_kfacseed_s$STEPS.eval.json $R/qadE6_kfacseed_s$STEPS.eval.log 1 || die "final score"
say "QAD20_DONE 20B IQ2_XXS QAD $(klk_of $R/qadE6_kfacseed_s$STEPS.eval.json) | K-FAC seed $(klk_of $REFJ) | llama.cpp PTQ $(klk_of $R/step1/E6.eval.json) | $G ($(du -h $G | cut -f1))"
