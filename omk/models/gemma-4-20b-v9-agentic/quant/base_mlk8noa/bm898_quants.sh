#!/bin/bash
# bm898_quants.sh <gpu> -- bm-898 plan step 3a (owner 2026-11-19 16:40 CET "go build the quants"): Q4_K_M + Q6_K of
# gemma-4-20B-v9-agentic-base-mlk8-noa-sft11 from the EXISTING F16 base (eos 106 / eot 1, built by
# bm898_f16_export_serve.sh) with the PRE-STAGED AtomicChat imatrix (gguf/imatrix.dat = hard link to imatrix_ac.dat;
# quantize_gguf 1081fc2 uses a pre-staged imatrix.dat as-is, no recompute, no --cal-data). Attention rule of that
# quantize_gguf: Q8_0 at Q4_K_M and above (both tiers here). --sanity-check loads each tier in llama-server and runs a
# completion (the sft12 Q4_K_M has the same .sanity.log). Plain llama-quantize tiers (PTQ) -- NOT the K-FAC rounder:
# the sft12 Q6_K eval column is a K-FAC Q6_K (mlk8-noa-sft12/gguf/Q6_K/kfac.gguf + ptq.gguf pair); a K-FAC Q6_K for
# sft11 is a separate factor pass, flagged to the owner.
# Refuses to overwrite an existing tier. Claim bs2/gpu0 first. Sentinels B898Q_START / B898Q_TIER_OK <tier> /
# B898Q_DONE / B898Q_FAIL.
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_quants.sh 0 > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/quants.chain.log 2>&1 < /dev/null &
set -uo pipefail
GPU=${1:?gpu index}
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11; BF=$O/$NAME; GG=$O/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11; OMK=/srv/ml/repos/omnimergekit; LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
TIERS="Q4_K_M Q6_K"
mkdir -p $O/logs; LOG=$O/logs/quants.driver.log
say() { echo "[b898q $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "B898Q_FAIL $*"; exit 1; }

# ---- preflight -------------------------------------------------------------------------------------------------
[ -s "$GG/$NAME-F16.gguf" ] || die "no F16 base $GG/$NAME-F16.gguf"
[ -s "$GG/imatrix.dat" ] && [ -s "$GG/imatrix_ac.dat" ] || die "no imatrix.dat / imatrix_ac.dat in $GG"
[ "$(stat -c %i "$GG/imatrix.dat")" = "$(stat -c %i "$GG/imatrix_ac.dat")" ] || die "imatrix.dat is not the hard link of imatrix_ac.dat"
sha256sum -c "$GG/imatrix_ac.dat.sha256" > /dev/null || die "imatrix_ac.dat sha mismatch"
[ -s "$TMPL" ] && [ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "chat template missing or sha != $TSHA"
for t in $TIERS; do [ -e "$GG/$NAME-$t.gguf" ] && die "$GG/$NAME-$t.gguf exists -- never overwritten"; done
[ -e "$O/logs/quants.out" ] && die "$O/logs/quants.out exists -- never overwritten"
free_g=$(df -BG --output=avail /mnt/sdc | tail -n 1 | tr -dc 0-9); [ "${free_g:-0}" -ge 120 ] || die "/mnt/sdc has ${free_g}G free (< 120G)"
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] || die "GPU$GPU not free (${used} MiB)"
cd "$OMK" && say "B898Q_START gpu $GPU tiers [$TIERS] F16 $(cut -c1-16 "$GG/$NAME-F16.gguf.sha256") imatrix_ac $(cut -c1-16 "$GG/imatrix_ac.dat.sha256") omk $(git log --oneline -n 1 | cut -c1-7) llama.cpp $LC" || die "cd $OMK"

# ---- quantize (one process, both tiers; F16 and imatrix reused) ------------------------------------------------
CUDA_VISIBLE_DEVICES=$GPU LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --only "$(echo $TIERS | tr ' ' ',')" --base-precision f16 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --ngl 99 --sanity-check --no-upload --keep-local \
  > "$O/logs/quants.out" 2>&1 < /dev/null || { tail -n 30 "$O/logs/quants.out" | tee -a $LOG; die "quantize_gguf"; }
grep -q "Using pre-staged imatrix.dat" "$O/logs/quants.out" || die "quantize_gguf did not use the pre-staged imatrix.dat"

# ---- verify each tier: sha, KV (imatrix file / chunks / experts / eos / file_type), attention tensor types ---------
for t in $TIERS; do
  g="$GG/$NAME-$t.gguf"; [ -s "$g" ] || die "no $g"
  sha256sum "$g" | tee "$g.sha256" | tee -a $LOG
  "$PY" - "$g" "$t" <<'PY' 2>&1 | tee -a $LOG || die "KV / tensor check $t"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); f = r.fields; tier = sys.argv[2]
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
im, ch, ex, eos, eot, ft = (get("quantize.imatrix.file"), get("quantize.imatrix.chunks_count"), get("gemma4.expert_count"),
                            get("tokenizer.ggml.eos_token_id"), get("tokenizer.ggml.eot_token_id"), get("general.file_type"))
types = {t.name: t.tensor_type.name for t in r.tensors}
attn = sorted({types[n] for n in types if ".attn_" in n and n.endswith(".weight") and "norm" not in n})
gu = sorted({types[n] for n in types if "ffn_gate_up_exps" in n or "ffn_up_exps" in n or "ffn_gate_exps" in n})
dn = sorted({types[n] for n in types if "ffn_down_exps" in n})
print(f"   {tier}: imatrix.file={im} chunks={ch} experts={ex} eos={eos} eot={eot} file_type={ft} tensors={len(types)}")
print(f"   {tier}: attention {attn}  routed gate_up {gu}  routed down {dn}")
assert im and ch == 9590 and ex == 98 and eos == 106 and eot == 1, "KV"
assert attn == ["Q8_0"], f"attention rule: expected Q8_0 at {tier}, got {attn}"
PY
  say "B898Q_TIER_OK $t $(stat -c %s "$g") B $g"
done
ls -la "$GG" | tee -a $LOG
say "B898Q_DONE $GG"
