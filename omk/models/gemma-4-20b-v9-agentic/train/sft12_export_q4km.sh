#!/bin/bash
# bm-893 (owner 2026-11-15 12:22 CET: "make a quick i-matrix, create a q4_k_m of the final sft12 before moving to the
# qad-gguf export score. install it into eleven2go as the others"). sft600_q4km.sh adapted to the sft12 run's FINAL
# checkpoint: wait for the trainer to exit with SFT12_OK + stage1/final.pt -> bf16 HF export (omk_qad --phase export
# --mode bf16, CPU, source = the mlk8-noa init) -> quick imatrix (calibration_datav5, all chunks) -> Q4_K_M, both on
# GPU1 (GPU0 runs the IQ2 QAD). Same imatrix/quant/KV steps as the other eleven2go tags (weights-only A/B).
# Never overwrites: refuses if the output dir exists; the export refuses a non-empty --out-hf.
# Sentinels: SFT12Q_WAIT / SFT12Q_STEP_OK <phase> / SFT12Q_DONE <gguf> / SFT12Q_FAIL. Usage: sft12_final_q4km.sh
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery
RUN=$G/runs/g4_bf16_sft12_98e_render_oc33; S1=$RUN/stage1; TLOG=$G/train_sft12_render_oc33.log
STEPS=855
TPY=/mnt/sdc/omk-model-optimizer/venv/bin/python
CODE=$G/omk_sft2_code_mlk8   # the tree sft12 trained with (sft12_train.sh PYTHONPATH)
M=/mnt/sdc/ml/v9a_sft11/mlk8-noa/gemma-4-20B-v9-agentic-sft11-mlk8-noa
O=/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12
NAME=gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12; BF=$O/$NAME; GG=$O/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/srv/ml/repos/omnimergekit
LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/v9a_sft11/omk/scripts/calibration_datav5.txt; CSHA=bd925aea03bec5dd
GPU=1
say() { echo "[sft12q $(date -u +%FT%TZ)] $*"; }
die() { say "SFT12Q_FAIL $*"; exit 1; }

[ -e "$O" ] && die "$O exists, refusing to overwrite"
[ -f "$M/BUILD_OK" ] || die "source $M not BUILD_OK"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ "$(sha256sum "$CALIB" | cut -c1-16)" = "$CSHA" ] || die "calib sha"
TP=$(cat $G/sft12_render_oc33_trainer.pid) || die "no trainer pid file"
mkdir -p "$O/logs" "$GG"

say "SFT12Q_WAIT for trainer $TP to exit with SFT12_OK and $S1/final.pt"
t0=$(date +%s)
while kill -0 "$TP" 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && die "trainer still running after 6h"
  sleep 60
done
sleep 20
grep -q "^SFT12_OK 2026-11-15T" $TLOG || die "trainer exited without SFT12_OK: $(grep -E 'SFT12_(FAIL|STOPPED_BY_RULE) 2026-11-15T' $TLOG | tail -1 | cut -c1-200) $(tail -2 $TLOG | tr '\n' ' ' | cut -c1-300)"
[ -s $S1/final.pt ] || die "no $S1/final.pt"
grep -q "\"step\": $STEPS" $S1/final.json 2>/dev/null || say "note: final.json step field: $(cat $S1/final.json 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
CK=$S1/final.pt
say "SFT12Q_STEP_OK ckpt $CK ($(stat -c %s $CK) B)"

free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 150 ] || die "/mnt/sdc ${free}G free < 150G (export 40G + F16 40G + Q4 13G)"
say "bf16 export (CPU) -> $BF"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE $TPY -m omk_qad.train --phase export --mode bf16 \
  --model $M --load $CK --out-hf $BF > $O/logs/export.log 2>&1 < /dev/null || { tail -20 $O/logs/export.log; die "export"; }
grep -q EXPORT_BF16 $O/logs/export.log || die "no EXPORT_BF16 line"
$PY - "$BF/omk_bf16_export.json" $STEPS <<'PY' || die "export report check"
import json, sys
r = json.load(open(sys.argv[1]))
print("[export] step", r["step"], "replaced", r["replaced"], "unchanged_verified", r["unchanged_verified"], "s", r["seconds"])
assert r["step"] == int(sys.argv[2]) and r["replaced"] > 0
PY
say "SFT12Q_STEP_OK export"

export CUDA_VISIBLE_DEVICES=$GPU
for i in $(seq 1 20); do
  used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 30
done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free (${used} MiB) 10 min after the trainer exited"
say "quick imatrix (calibration_datav5, all chunks) on GPU$GPU"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --imatrix-only \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > "$O/logs/imatrix.out" 2>&1 < /dev/null || { tail -30 "$O/logs/imatrix.out"; die "imatrix"; }
grep -q IMATRIX_ONLY_DONE "$O/logs/imatrix.out" || die "no IMATRIX_ONLY_DONE marker"
"$PY" - "$GG/imatrix.dat" <<'PY' || die "imatrix check"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["imatrix.chunk_count"]; print("[imatrix] chunk_count=%d entries=%d" % (f.parts[f.data[0]].tolist()[0], len(r.tensors)))
PY
sha256sum "$GG/imatrix.dat" | tee "$GG/imatrix.dat.sha256"
say "SFT12Q_STEP_OK imatrix"

say "Q4_K_M with the quick imatrix on GPU$GPU"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --only Q4_K_M \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --ngl 99 --no-upload --keep-local \
  > "$O/logs/quant.out" 2>&1 < /dev/null || { tail -30 "$O/logs/quant.out"; die "quant"; }
g=$(ls "$GG"/*Q4_K_M.gguf 2>/dev/null | head -1)
[ -s "$g" ] || die "no Q4_K_M GGUF"
sha256sum "$g" | tee "$g.sha256"
"$PY" - "$g" <<'PY' || die "GGUF KV check"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
im, ex, eos = get("quantize.imatrix.file"), get("gemma4.expert_count"), get("tokenizer.ggml.eos_token_id")
print(f"   imatrix.file={im} chunks={get('quantize.imatrix.chunks_count')} experts={ex} eos={eos}")
assert im and ex == 98 and eos == 106
PY
ls -la "$GG"
grep -m1 '"tag": "final"\|"tag": "step855"' $TLOG | cut -c1-400
say "SFT12Q_DONE $g"
