#!/bin/bash
# relocate_26b_artifacts_bs2.sh -- RUNS ON bs2 (owner 2026-11-20 06:54 CET: "move 26b_agentic_kfac/round and
# 26b_agentic_iq2_r16k in the public 26b repo"). The Hub has no cross-repo copy, so each folder is downloaded to the
# sdc lane, uploaded to the PUBLIC ManniX-ITA/gemma-4-26B-A4B-agentic under omk-artifacts/<folder> with hf_move_dir.py
# (sha-verified file by file, local big files deleted only after the hub sha matches), then deleted from the private
# artifacts repo, then the local folder removed. Sequential, one folder at a time (47.5 + 63.1 GiB), 150G sdc floor.
# Ends with a squash of the artifacts repo. Sentinels REL_START / REL_PART <d> / REL_FAIL / REL_DONE.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer/runs; W=$R/hf_relocate_26b; L=$R/relocate_26b.log
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11; HF=/srv/ml/envs/envs/omnimergekit/bin/hf
MVPY=/srv/ml/envs/envs/omk-modelopt/bin/python
SRC=ManniX-ITA/omk-qad-artifacts-gemma4-agentic; DST=ManniX-ITA/gemma-4-26B-A4B-agentic
say() { echo "[rel $(date -u +%FT%TZ)] $*" | tee -a "$L"; }
die() { say "REL_FAIL $*"; exit 1; }
[ -x "$HF" ] || die "no hf CLI at $HF"
[ -s "$R/hf_move_dir.py" ] || die "no hf_move_dir.py"
mkdir -p "$W"
say "REL_START sdc $(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)G free"
for d in 26b_agentic_kfac/round 26b_agentic_iq2_r16k; do
  avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 150 ] || die "sdc ${avail}G < 150G floor before $d"
  say "download $d"
  "$HF" download "$SRC" --include "$d/*" --local-dir "$W" > "$W/dl_$(basename "$d").log" 2>&1 \
    || { tail -n 5 "$W/dl_$(basename "$d").log" | tee -a "$L"; die "download $d"; }
  n=$(find "$W/$d" -type f | wc -l); say "downloaded $d: $n files $(du -sh "$W/$d" | cut -f1)"
  say "upload $d -> $DST/omk-artifacts/$d"
  out=$("$MVPY" "$R/hf_move_dir.py" "$DST" "$W/$d" "omk-artifacts/$d" 2>&1) \
    || { echo "$out" | tail -n 5 | tee -a "$L"; die "upload $d"; }
  say "$(echo "$out" | grep MV_OK)"
  say "delete $d from $SRC"
  "$PY" - "$SRC" "$d" <<'EOF' || die "hub delete $d"
import sys
from huggingface_hub import HfApi
api = HfApi()
repo, d = sys.argv[1], sys.argv[2]
api.delete_folder(path_in_repo=d, repo_id=repo, commit_message=f"{d}: moved to the public 26B repo (omk-artifacts/{d})")
print("deleted", d)
EOF
  rm -rf "${W:?}/${d:?}"
  say "REL_PART $d"
done
"$PY" - "$SRC" <<'EOF' || die "squash"
import sys
from huggingface_hub import HfApi
api = HfApi()
r = sys.argv[1]
n = len(api.list_repo_commits(r))
api.super_squash_history(r, commit_message="artifacts: 26B folders moved to the public 26B repo (history squashed 2026-11-20)")
print("squashed", n, "->", len(api.list_repo_commits(r)))
EOF
say "REL_DONE sdc $(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)G free"
