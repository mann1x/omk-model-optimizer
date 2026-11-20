#!/bin/bash
# bm898_ac_imatrix.sh -- owner 2026-11-19 14:05 CET: "don't build the quick i-matrix, build directly the full AC i-matrix for
# v9-agentic sft11, use the bs2 gpu0". = sft12_ac_imatrix.sh (bm-893) for the bm-898 model gemma-4-20B-v9-agentic-base-mlk8-noa-sft11:
# the official v9 AtomicChat corpus /mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt, ALL chunks, --parse-special
# (quantize_gguf default). The F16 already exists in the gguf dir (bm898_f16_export_serve.sh built it, eos 106 / eot 1), so
# quantize_gguf reuses it ("Base GGUF exists") and no second 37 GB copy is written. The result is kept under BOTH names:
# imatrix_ac.dat (+sha, the sft12 convention) and imatrix.dat as a hard link to it (what quantize_gguf picks up for the tiers).
# No quick (calibration_datav5) imatrix exists for this model, by the owner's word. Claim bs2/gpu0 first.
# Usage: bm898_ac_imatrix.sh <gpu>     Sentinels ACIM_START / ACIM_DONE / ACIM_FAIL
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_ac_imatrix.sh 0 > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/ac_imatrix.chain.log 2>&1 < /dev/null &
set -uo pipefail
GPU=${1:?gpu index}
O=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11; BF=$O/$NAME; GG=$O/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11; OMK=/srv/ml/repos/omnimergekit; LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt
mkdir -p $O/logs; LOG=$O/logs/ac_imatrix.driver.log
say() { echo "[acim $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "ACIM_FAIL $*"; exit 1; }
[ -d $BF ] || die "no bf16 model $BF"
[ -s $GG/$NAME-F16.gguf ] || die "no F16 $GG/$NAME-F16.gguf"
[ -e $GG/$NAME-F16.gguf.eog.tmp ] && die "EOG rewrite still in progress ($GG/$NAME-F16.gguf.eog.tmp)"
[ -s $CALIB ] || die "no AC corpus"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ -e $GG/imatrix_ac.dat ] && die "$GG/imatrix_ac.dat exists (never overwritten)"
[ -e $GG/imatrix.dat ] && die "$GG/imatrix.dat exists (never overwritten; a partial quick one must be renamed first)"
"$PY" - "$GG/$NAME-F16.gguf" <<'PY' || die "F16 KV check"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
ex, eos, eot, ft = get("gemma4.expert_count"), get("tokenizer.ggml.eos_token_id"), get("tokenizer.ggml.eot_token_id"), get("general.file_type")
print(f"   F16: experts={ex} eos={eos} eot={eot} file_type={ft}")
assert ex == 98 and eos == 106 and eot == 1 and ft == 1
PY
say "ACIM_START gpu $GPU corpus $(wc -c < "$CALIB") bytes sha $(sha256sum "$CALIB" | cut -c1-16); F16 reused from $GG"
export CUDA_VISIBLE_DEVICES=$GPU
for _ in $(seq 1 240); do used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] && break; sleep 30; done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU busy ($used MiB) for 2h"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --imatrix-only --base-precision f16 \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > $O/logs/ac_imatrix.out 2>&1 < /dev/null || { tail -30 $O/logs/ac_imatrix.out | tee -a $LOG; die "imatrix"; }
grep -q IMATRIX_ONLY_DONE $O/logs/ac_imatrix.out || die "no IMATRIX_ONLY_DONE marker"
[ -s $GG/imatrix.dat ] || die "no $GG/imatrix.dat"
mv $GG/imatrix.dat $GG/imatrix_ac.dat && ln $GG/imatrix_ac.dat $GG/imatrix.dat
sha256sum $GG/imatrix_ac.dat > $GG/imatrix_ac.dat.sha256
cp $O/logs/ac_imatrix.out $GG/imatrix_ac_run.log
printf '%s\n' "imatrix_ac.dat = the full AtomicChat (ac_gemma4_v9official calib_train.txt, all chunks) imatrix of $NAME, built $(date -u +%FT%TZ) on bs2 GPU$GPU." "imatrix.dat is a hard link to it (same bytes). No quick calibration_datav5 imatrix exists for this model (owner 2026-11-19 14:05)." > $GG/IMATRIX_README.txt
"$PY" - $GG/imatrix_ac.dat <<'PY' | tee -a $LOG || die "imatrix check"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
kv = {k: v for k, v in ((f.name, f) for f in r.fields.values())}
n = sum(1 for t in r.tensors if t.name.endswith(".in_sum2"))
chunks = kv.get("imatrix.chunk_count"); ds = kv.get("imatrix.datasets")
print("imatrix_ac.dat: %d in_sum2 tensors, chunk_count %s, datasets %s" % (
    n, chunks.parts[chunks.data[0]].tolist() if chunks else "?", bytes(ds.parts[ds.data[0]]).decode()[:80] if ds else "?"))
PY
say "ACIM_DONE $GG/imatrix_ac.dat $(stat -c %s $GG/imatrix_ac.dat) B (imatrix.dat hard-linked)"
