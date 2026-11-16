#!/bin/bash
# bm-893 (owner 2026-11-16 10:34): the AtomicChat imatrix for gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12.
# The existing gguf/imatrix.dat is from calibration_datav5 (quick build); this one uses the official v9 AC corpus
# /mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt, ALL chunks, --parse-special (quantize_gguf default).
# The F16 (39 GB) is rebuilt on the ROOT fs (/mnt/sdc holds the 26B retrain; / has 244 GB, floor 200) and reported,
# not deleted. imatrix.dat is copied to the sft12 gguf dir as imatrix_ac.dat (+sha) -- an imatrix is never deleted.
# Usage: sft12_ac_imatrix.sh <gpu>     Sentinels ACIM_START / ACIM_DONE / ACIM_FAIL
set -uo pipefail
GPU=${1:?gpu index}
O=/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12; NAME=gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12; BF=$O/$NAME; GG=$O/gguf
WORK=/srv/ml/scratch/sft12_ac_imatrix
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11; OMK=/srv/ml/repos/omnimergekit; LC=/mnt/sdc/ml/llama.cpp-b12386
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt
mkdir -p $WORK $O/logs; LOG=$O/logs/ac_imatrix.driver.log
say() { echo "[acim $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "ACIM_FAIL $*"; exit 1; }
[ -d $BF ] || die "no bf16 model $BF"
[ -s $CALIB ] || die "no AC corpus"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ -e $GG/imatrix_ac.dat ] && die "$GG/imatrix_ac.dat exists (never overwritten)"
free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "${free:-0}" -ge 240 ] || die "root fs ${free}G free: the 39 GB F16 would breach the 200 GB floor"
say "ACIM_START gpu $GPU corpus $(wc -c < $CALIB) bytes sha $(sha256sum $CALIB | cut -c1-16); F16 on $WORK"
export CUDA_VISIBLE_DEVICES=$GPU
for i in $(seq 1 240); do used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] && break; sleep 30; done
[ "${used:-99999}" -lt 2000 ] || die "GPU$GPU busy ($used MiB) for 2h"
LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$WORK" --imatrix-only \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > $O/logs/ac_imatrix.out 2>&1 < /dev/null || { tail -30 $O/logs/ac_imatrix.out | tee -a $LOG; die "imatrix"; }
grep -q IMATRIX_ONLY_DONE $O/logs/ac_imatrix.out || die "no IMATRIX_ONLY_DONE marker"
IM=$(ls $WORK/imatrix.dat $WORK/*/imatrix.dat 2>/dev/null | head -1); [ -s "$IM" ] || die "no imatrix.dat under $WORK"
cp "$IM" $GG/imatrix_ac.dat && sha256sum $GG/imatrix_ac.dat > $GG/imatrix_ac.dat.sha256
cp $O/logs/ac_imatrix.out $GG/imatrix_ac_run.log
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
say "ACIM_DONE $GG/imatrix_ac.dat $(stat -c %s $GG/imatrix_ac.dat) B; F16 left at $(ls $WORK/*.gguf 2>/dev/null | head -1) ($(du -sh $WORK | cut -f1)) on the root fs -- report, do not delete without the owner"
