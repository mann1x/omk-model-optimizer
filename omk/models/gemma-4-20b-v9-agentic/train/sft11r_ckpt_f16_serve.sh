#!/bin/bash
# sft11r_ckpt_f16_serve.sh STEP -- owner 2026-11-20 16:15 CET: "step 300 now, then we see if to swap for 400, yes take care of the
# per-expert scale during conversion; replace the model loaded now on 8244". The bm-898 sft11r run (bf16 storage, relative fp
# lr 3e-5, RUN_TAG bf16st_rlr3e5) is still training on GPU0; this serves one of its numbered snapshots on GPU1:
#   1. hard-link stage1/ckpt_step<STEP>.pt out of the snapshot rotation, bf16 HF export on the CPU (omk_qad --phase export,
#      the run's own code tree) -> the 295 matrices AND the 210 trained scale tensors must be written (505)
#   (v2 2026-11-20: export keeps the scales fp32 -- the bf16 cast erased them; no imatrix)
#   2. stop the 8244 server (the base-mlk8-noa-sft11 F16; its GGUF stays on disk), F16 GGUF with eos 106 / eot 1 via the canonical
#      quantize_gguf.py --imatrix-only (as bm898_f16_export_serve.sh)
#   3. SCALE CHECK on the GGUF: blk.N.ffn_gate_inp.scale (router.scale) and blk.N.ffn_down_exps.scale (per_expert_scale) must
#      exist for all 30 layers and DIFFER from the sft11 F16 (vendor values) -- llama.cpp b12386 conversion/gemma.py maps both
#      and src/models/gemma4.cpp applies them, so no fold into down_proj is needed; this proves the trained values arrived
#   4. start 8244 with the same recipe (alias gemma-4-20b-v9-agentic-sft11r-s<STEP>-f16, matches `gemma*`) and smoke it
# Never overwrites: refuses if the output dir exists. bs2/gpu1 is claimed from solidpc.
# Sentinels: S11RF_STEP_OK <phase> / S11RF_SERVE_OK / S11RF_DONE / S11RF_FAIL.
set -uo pipefail
STEP=${1:?step}; S7=$(printf %07d "$STEP")
G=/mnt/sdc/omk-model-optimizer/g4_recovery
S1=$G/runs/g4_bf16_sft11r_base_mlk8noa_bf16st_rlr3e5/stage1
TPY=/mnt/sdc/omk-model-optimizer/venv/bin/python
CODE=$G/omk_sft11r_export_code   # the run's tree + the fp32-scale export fix (recovery.py); the trainer's own tree is not touched
M=/mnt/sdc/ml/v9a_base/mlk8-noa/gemma-4-20B-v9-agentic-base-mlk8-noa
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11r-s$STEP
NAME=gemma-4-20B-v9-agentic-sft11r-s$STEP; BF=$O/$NAME; GG=$O/gguf
REF=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-F16.gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/srv/ml/repos/omnimergekit
LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/v9a_sft11/omk/scripts/calibration_datav5.txt; CSHA=bd925aea03bec5dd
GPU=1
OLD=/srv/ml/start-8244-g4-20b-base-mlk8noa-sft11-f16-f16kv.sh
SERVE=/srv/ml/start-8244-g4-20b-sft11r-s$STEP-f16-f16kv.sh
SLOG=/srv/ml/opencoti-dev/logs/start-8244-g4-20b-sft11r-s$STEP-f16-f16kv.log
ALIAS=gemma-4-20b-v9-agentic-sft11r-s$STEP-f16
say() { echo "[s11rf $(date -u +%FT%TZ)] $*"; }
die() { say "S11RF_FAIL $*"; exit 1; }

[ -e "$O" ] && die "$O exists, refusing to overwrite"
[ -s $S1/ckpt_step$S7.pt ] && [ -s $S1/ckpt_step$S7.json ] || die "no $S1/ckpt_step$S7.pt"
[ -f "$M/BUILD_OK" ] || die "source $M not BUILD_OK"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ "$(sha256sum "$CALIB" | cut -c1-16)" = "$CSHA" ] || die "calib sha"
[ -x "$OLD" ] && [ -s "$REF" ] || die "no reference server script / F16"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 200 ] || die "/mnt/sdc ${free}G free < 200G"
mkdir -p "$O/logs" "$GG"
ln $S1/ckpt_step$S7.pt $S1/ckpt_step$S7.json "$O/" || die "hard link of the snapshot"
say "S11RF_STEP_OK preflight (free ${free}G, snapshot linked)"

say "bf16 export (CPU) -> $BF"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE $TPY -m omk_qad.train --phase export --mode bf16 \
  --model $M --load "$O/ckpt_step$S7.pt" --out-hf $BF > $O/logs/export.log 2>&1 < /dev/null || { tail -20 $O/logs/export.log; die "export"; }
grep -q EXPORT_BF16 $O/logs/export.log || die "no EXPORT_BF16 line"
$PY - "$BF/omk_bf16_export.json" "$STEP" <<'PY' || die "export report check"
import json, sys
r = json.load(open(sys.argv[1]))
k = r["replaced_keys"]
sc = [x for x in k if x.endswith("router.scale")]; pe = [x for x in k if x.endswith("per_expert_scale")]
nm = [x for x in k if "feedforward_layernorm" in x]
print("[export] step", r["step"], "replaced", r["replaced"], "router.scale", len(sc), "per_expert_scale", len(pe), "ffn norms", len(nm),
      "unchanged_verified", r["unchanged_verified"], "s", r["seconds"])
assert r["step"] == int(sys.argv[2]) and r["replaced"] == 505 and len(sc) == 30 and len(pe) == 30 and len(nm) == 150
assert len(r["fp32_keys"]) == 210, len(r.get("fp32_keys", []))
print("[export] fp32 scale tensors", len(r["fp32_keys"]))
PY
say "S11RF_STEP_OK export"

SP0=$(ss -ltnp 2>/dev/null | grep "127.0.0.1:8244 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
if [ -n "${SP0:-}" ]; then
  tr '\0' ' ' < /proc/$SP0/cmdline | grep -q "base-mlk8-noa-sft11-F16.gguf" || die "8244 is held by an unexpected process $SP0"
  say "stopping the 8244 server pid $SP0 (sft11 F16; GGUF kept)"
  kill -TERM "$SP0"; for _ in $(seq 1 60); do kill -0 "$SP0" 2>/dev/null || break; sleep 2; done
  kill -0 "$SP0" 2>/dev/null && die "8244 server $SP0 did not exit"
fi
export CUDA_VISIBLE_DEVICES=$GPU
for _ in $(seq 1 40); do used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] && break; sleep 5; done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free (${used} MiB)"
say "F16 base GGUF (eos 106), no imatrix (serve-only build)"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --no-imatrix --imatrix-only --base-precision f16 \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > "$O/logs/f16_base.out" 2>&1 < /dev/null
# --no-imatrix + --imatrix-only = build the base and stop: the tool ends with this exact line and exit 1, by design here
grep -q "imatrix-only: FAILED — no imatrix was produced" "$O/logs/f16_base.out" || { tail -30 "$O/logs/f16_base.out"; die "f16 base"; }
F16=$GG/$NAME-F16.gguf
[ -s "$F16" ] || die "no $F16"
"$PY" - "$F16" "$REF" <<'PY' || die "F16 GGUF KV / scale check"
import sys
import numpy as np
from gguf import GGUFReader
new, ref = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
f = new.fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
ex, eos, eot, ft = get("gemma4.expert_count"), get("tokenizer.ggml.eos_token_id"), get("tokenizer.ggml.eot_token_id"), get("general.file_type")
print(f"   experts={ex} eos={eos} eot={eot} file_type={ft}")
assert ex == 98 and eos == 106 and eot == 1 and ft == 1
tn = {t.name: t for t in new.tensors}; tr = {t.name: t for t in ref.tensors}
assert set(tn) == set(tr), (sorted(set(tn) ^ set(tr))[:6])
for suffix, what in ((".ffn_gate_inp.scale", "router.scale"), (".ffn_down_exps.scale", "per_expert_scale")):
    names = sorted(n for n in tn if n.endswith(suffix))
    assert len(names) == 30, (suffix, len(names))
    diff, rel = 0, []
    for n in names:
        a, b = np.asarray(tn[n].data, dtype=np.float64), np.asarray(tr[n].data, dtype=np.float64)
        assert a.shape == b.shape
        if not np.array_equal(a, b):
            diff += 1
        rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    print(f"   {what:17s} -> blk.N{suffix}: 30 tensors, {diff} differ from the sft11 F16, dtype {tn[names[0]].tensor_type.name}, "
          f"relative change mean {np.mean(rel):.2e} max {np.max(rel):.2e}")
    assert diff == 30, f"{what}: only {diff}/30 layers carry trained values"
one = [n for n in tn if len(tn[n].shape) == 1]
d1 = sum(1 for n in one if not np.array_equal(np.asarray(tn[n].data), np.asarray(tr[n].data)))
print(f"   1-D tensors: {len(one)} in the GGUF, {d1} differ from the sft11 F16 (210 trained: 60 router + 150 FFN norms)")
assert d1 == 210, d1
PY
sha256sum "$F16" | tee "$F16.sha256"
say "S11RF_STEP_OK f16 $F16"

sed -e "s|^M=.*|M=$F16|" -e "s|-a gemma-4-20b-v9-agentic-base-mlk8noa-sft11-f16|-a $ALIAS|" "$OLD" > "$SERVE" && chmod +x "$SERVE"
grep -q "^M=$F16\$" "$SERVE" && grep -q -- "-a $ALIAS " "$SERVE" || die "server script rewrite"
for _ in $(seq 1 20); do used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] && break; sleep 15; done
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
curl -s http://127.0.0.1:8244/v1/models | grep -q "\"$ALIAS\"" || die "alias $ALIAS not served"
curl -s http://127.0.0.1:8244/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":64,\"temperature\":0}" > "$O/logs/smoke.json" 2>&1
grep -q '"content"' "$O/logs/smoke.json" || { cut -c1-400 "$O/logs/smoke.json"; die "smoke completion"; }
say "smoke: $(cut -c1-300 "$O/logs/smoke.json")"
echo "$SP" > "$O/server.pid"
say "S11RF_SERVE_OK pid $SP alias $ALIAS http://127.0.0.1:8244"
say "S11RF_DONE $F16"
