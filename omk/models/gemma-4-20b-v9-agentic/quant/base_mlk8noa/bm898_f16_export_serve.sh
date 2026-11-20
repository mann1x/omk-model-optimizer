#!/bin/bash
# bm898_f16_export_serve.sh -- owner 2026-11-19 13:55 CET: "serve the 20b v9-agentic sft11 f16 on bs2 gpu1 with the 8244 server
# so I can test it and find the sampling parameters" (+ 13:58: the alias must match Cerebriline's gemma template `gemma*`).
# The bm-898 model (gemma-4-20B-v9-agentic-base-mlk8-noa + sft11, SFTB_OK 2026-11-19T12:48:26Z, 2108 steps):
#   1. bf16 HF export of stage1/final.pt (omk_qad --phase export --mode bf16, CPU, source = the pruned base it trained from,
#      code tree = the one it trained with) -> /mnt/sdc/ml/v9a_base/mlk8-noa-sft11/<NAME>
#   2. F16 base GGUF with the Gemma-4 eos 106 / eot 1 normalisation via the canonical quantize_gguf.py --imatrix-only
#      (the quick-pipeline step that built the base's F16 on 11-18; the imatrix it also writes is the one step 3 of the plan
#      needs for Q4_K_M / Q6_K, so nothing is computed twice) -> <OUT>/gguf/<NAME>-F16.gguf (+ imatrix.dat)
#   3. start the 8244 server (start-8244-g4-20b-base-mlk8noa-sft11-f16-f16kv.sh = the 26B bm-889 8244 bf16/f16-KV recipe with
#      this model, GPU1, alias gemma-4-20b-v9-agentic-base-mlk8noa-sft11-f16, the 20B sampler presence 0.035) and smoke it.
# Never overwrites: refuses if OUT exists. bs2/gpu1 is claimed from solidpc before this starts.
# Sentinels: B898F_STEP_OK <phase> / B898F_SERVE_OK / B898F_DONE / B898F_FAIL.
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_f16_export_serve.sh > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/chain.log 2>&1 < /dev/null &
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery
S1=$G/runs/g4_bf16_sft11_base_mlk8noa/stage1; TLOG=$G/train_sft11_base_mlk8noa.log
STEPS=2108
TPY=/mnt/sdc/omk-model-optimizer/venv/bin/python
CODE=$G/omk_sft2_code_mlk8   # the tree bm-898 trained with (sft11_base_mlk8noa_train.sh PYTHONPATH)
M=/mnt/sdc/ml/v9a_base/mlk8-noa/gemma-4-20B-v9-agentic-base-mlk8-noa
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11
NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11; BF=$O/$NAME; GG=$O/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/srv/ml/repos/omnimergekit
LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/v9a_sft11/omk/scripts/calibration_datav5.txt; CSHA=bd925aea03bec5dd
GPU=1
SERVE=/srv/ml/start-8244-g4-20b-base-mlk8noa-sft11-f16-f16kv.sh
SLOG=/srv/ml/opencoti-dev/logs/start-8244-g4-20b-base-mlk8noa-sft11-f16-f16kv.log
ALIAS=gemma-4-20b-v9-agentic-base-mlk8noa-sft11-f16
say() { echo "[b898f $(date -u +%FT%TZ)] $*"; }
die() { say "B898F_FAIL $*"; exit 1; }

[ -e "$O" ] && die "$O exists, refusing to overwrite"
[ -f "$M/BUILD_OK" ] || die "source $M not BUILD_OK"
[ "$(cat "$(dirname "$M")/shared_alpha.txt" 2>/dev/null)" = none ] || die "source is not the no-α build"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ "$(sha256sum "$CALIB" | cut -c1-16)" = "$CSHA" ] || die "calib sha"
[ -x "$SERVE" ] || die "no server script $SERVE"
grep -q "^SFTB_OK 2026-11-19T" $TLOG || die "no SFTB_OK in $TLOG"
[ -s $S1/final.pt ] || die "no $S1/final.pt"
grep -q "\"step\": $STEPS" $S1/final.json 2>/dev/null || say "note: final.json: $(tr '\n' ' ' < $S1/final.json | cut -c1-200)"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 150 ] || die "/mnt/sdc ${free}G free < 150G (export 40G + F16 40G, keep >= 200G after)"
ss -ltn | grep -q "127.0.0.1:8244 " && die "something already listens on 127.0.0.1:8244"
mkdir -p "$O/logs" "$GG"
say "B898F_STEP_OK preflight (free ${free}G)"

say "bf16 export (CPU) -> $BF"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE $TPY -m omk_qad.train --phase export --mode bf16 \
  --model $M --load $S1/final.pt --out-hf $BF > $O/logs/export.log 2>&1 < /dev/null || { tail -20 $O/logs/export.log; die "export"; }
grep -q EXPORT_BF16 $O/logs/export.log || die "no EXPORT_BF16 line"
$PY - "$BF/omk_bf16_export.json" $STEPS <<'PY' || die "export report check"
import json, sys
r = json.load(open(sys.argv[1]))
print("[export] step", r["step"], "replaced", r["replaced"], "unchanged_verified", r["unchanged_verified"], "s", r["seconds"])
assert r["step"] == int(sys.argv[2]) and r["replaced"] > 0
PY
say "B898F_STEP_OK export"

export CUDA_VISIBLE_DEVICES=$GPU
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free (${used} MiB)"
say "F16 base GGUF (eos 106) + quick imatrix (calibration_datav5, all chunks) on GPU$GPU"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --imatrix-only --base-precision f16 \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > "$O/logs/f16_imatrix.out" 2>&1 < /dev/null || { tail -30 "$O/logs/f16_imatrix.out"; die "f16/imatrix"; }
grep -q IMATRIX_ONLY_DONE "$O/logs/f16_imatrix.out" || die "no IMATRIX_ONLY_DONE marker"
F16=$GG/$NAME-F16.gguf
[ -s "$F16" ] || die "no $F16"
"$PY" - "$F16" <<'PY' || die "F16 GGUF KV check"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
ex, eos, eot, ft = get("gemma4.expert_count"), get("tokenizer.ggml.eos_token_id"), get("tokenizer.ggml.eot_token_id"), get("general.file_type")
print(f"   experts={ex} eos={eos} eot={eot} file_type={ft}")
assert ex == 98 and eos == 106 and eot == 1 and ft == 1  # 1 = MOSTLY_F16
PY
sha256sum "$F16" | tee "$F16.sha256"
sha256sum "$GG/imatrix.dat" | tee "$GG/imatrix.dat.sha256"
say "B898F_STEP_OK f16 $F16"

for _ in $(seq 1 20); do
  used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 15
done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free (${used} MiB) after the imatrix"
say "starting the 8244 server: $SERVE"
setsid nohup bash "$SERVE" >> "$SLOG" 2>&1 < /dev/null &
SP=$!
for _ in $(seq 1 120); do
  sleep 5
  kill -0 $SP 2>/dev/null || { tail -20 "$SLOG"; die "server exited during load"; }
  curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8244/health 2>/dev/null | grep -q 200 && break
done
curl -s http://127.0.0.1:8244/health | grep -q '"ok"' || { tail -20 "$SLOG"; die "server not healthy after 10 min"; }
curl -s http://127.0.0.1:8244/v1/models | grep -q "\"$ALIAS\"" || die "alias $ALIAS not served: $(curl -s http://127.0.0.1:8244/v1/models | cut -c1-300)"
curl -s http://127.0.0.1:8244/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":64,\"temperature\":0}" > "$O/logs/smoke.json" 2>&1
grep -q '"content"' "$O/logs/smoke.json" || { cut -c1-400 "$O/logs/smoke.json"; die "smoke completion"; }
say "smoke: $(cut -c1-300 "$O/logs/smoke.json")"
say "B898F_SERVE_OK pid $SP alias $ALIAS http://127.0.0.1:8244 (docker bridge 172.17.0.1:8244 via socat)"
say "B898F_DONE $F16"
