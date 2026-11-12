#!/bin/bash
# agentic_drive.sh -- solidpc driver for agentic_bs2.sh (owner 2026-11-12 17:59 CET). Ships the AC corpus (re-rendered on
#   solidpc, sha 33fef70c1f8d5965) and the canonical omk quantize_gguf.py snapshot to bs2, runs export (CPU), then in
#   parallel: upload (network) and imatrix (bs2/gpu0, claimed AT POINT OF USE). Archives imatrix.dat to solidpc.
#   Every wait has a failure exit. Sentinels AGENTIC_DRIVE_OK / AGENTIC_DRIVE_FAIL.
set -uo pipefail
H=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/scripts/bm889_sft6/sft7/v11
G=/mnt/sdc/omk-model-optimizer/g4_recovery
CC=/shared/dev/calib-corpora/builds/gemma-4-20b-v9-agentic
BC=/mnt/sdc/ml/calib/ac_gemma4_v9official
OMKS=/mnt/sdc/ml/omk_snap_54ea808/scripts
GG=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic-GGUF
ARC=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/gemma-4-26B-A4B-agentic
CL=""
rs(){ ssh -o ConnectTimeout=30 bs2 "$1" < /dev/null 2>&1; }
release_all(){ for r in $CL; do claude-hooks-res release $r 2>&1 | tail -1; done; CL=""; }
fail(){ echo "AGENTIC_DRIVE_FAIL $(date -u +%FT%TZ) $*"; release_all; exit 1; }
launch(){ rs "test -e $G/agentic_$1.out && echo EXISTS" | grep -q EXISTS && fail "agentic_$1.out exists"
  rs "setsid nohup bash $G/agentic_bs2.sh $1 > $G/agentic_$1.out 2>&1 < /dev/null &" > /dev/null; }
# wait STAGE HOURS: until the sentinel; fail on FAIL, on a dead script without a sentinel, or on the deadline
wait_st(){ local U; U=$(echo "$1" | tr a-z A-Z); local end=$(( $(date +%s) + $2*3600 ))
  while :; do sleep 60
    o=$(rs "grep -E '^AGENTIC_${U}_(OK|FAIL)' $G/agentic_$1.out; ps -eo pid,args | grep -c '[a]gentic_bs2.sh $1'")
    echo "$o" | grep -q "^AGENTIC_${U}_OK" && return 0
    echo "$o" | grep -q "^AGENTIC_${U}_FAIL" && { rs "tail -n 25 $G/agentic_$1.out"; return 1; }
    [ "$(echo "$o" | tail -1)" = 0 ] && { rs "tail -n 25 $G/agentic_$1.out"; echo "exited without a sentinel"; return 1; }
    [ "$(date +%s)" -gt $end ] && { echo "still running after $2 h"; return 1; }
  done; }

# 0 ship corpus + omk snapshot + stage script (new dirs only)
[ "$(sha256sum $CC/calib_train.txt | cut -c1-16)" = 33fef70c1f8d5965 ] || fail "local corpus sha"
rs "test -e $BC && echo EXISTS" | grep -q EXISTS || { rs "mkdir -p $BC"; scp -q $CC/calib_train.txt $CC/calib_train.manifest.jsonl $CC/manifest.json bs2:$BC/ || fail "scp corpus"; }
[ "$(rs "sha256sum $BC/calib_train.txt" | cut -c1-16)" = 33fef70c1f8d5965 ] || fail "bs2 corpus sha"
rs "test -e $OMKS && echo EXISTS" | grep -q EXISTS || { rs "mkdir -p $OMKS"; scp -q /shared/dev/omnimergekit/scripts/quantize_gguf.py /shared/dev/omnimergekit/scripts/gguf_kld.py bs2:$OMKS/ || fail "scp omk"; }
scp -q $H/agentic_bs2.sh bs2:$G/agentic_bs2.sh || fail "scp stage script"
echo "SHIPPED $(date -u +%FT%TZ) corpus 33fef70c, omk 54ea808 snapshot"

# 1 export (CPU)
launch export; wait_st export 2 || fail "export"; rs "grep -E '^(>>>|EXPORT_CHECK|AGENTIC)' $G/agentic_export.out"

# 2 upload in the background on bs2, imatrix on GPU0 (claim at point of use)
launch upload
dl=$(( $(date +%s) + 6*3600 ))
while :; do
  out=$(claude-hooks-res acquire bs2/gpu0:exclusive -r "bm-889 agentic: full AtomicChat imatrix of sft11-final on GPU0 (owner 2026-11-12 17:59)" --ttl 12h 2>&1) && break
  echo "$out" | grep -qiE 'held|busy|holder|waiting' || fail "acquire bs2/gpu0 failed (not busy): $out"
  echo "$(date -u +%FT%TZ) gpu0 busy: $(echo "$out" | head -1 | cut -c1-160)"; [ "$(date +%s)" -gt "$dl" ] && fail "gpu0 held 6 h"; sleep 120
done
CL=" bs2/gpu0"; echo "CLAIM_OK $(date -u +%FT%TZ) $out"
launch imatrix

wait_st upload 6 && echo "UPLOAD_DONE $(rs "grep -E '^(SHA_OK|REPO|AGENTIC_UPLOAD)' $G/agentic_upload.out" | tr '\n' ' ' | cut -c1-900)" || { echo "UPLOAD_FAILED (imatrix continues)"; UPF=1; }
wait_st imatrix 30 || fail "imatrix"
release_all

# 3 archive imatrix to solidpc (MANDATORY: imatrix preserved next to the quants AND on solidpc)
mkdir -p $ARC; [ -e $ARC/imatrix.dat ] && fail "$ARC/imatrix.dat exists"
scp -q "bs2:$GG/imatrix*.dat" $ARC/ || fail "scp imatrix"
rs "cd $GG && sha256sum imatrix*.dat" > $ARC/imatrix.sha256.bs2; (cd $ARC && sha256sum imatrix*.dat) > $ARC/imatrix.sha256
diff <(awk '{print $1, $2}' $ARC/imatrix.sha256 | sort) <(awk '{print $1, $2}' $ARC/imatrix.sha256.bs2 | sort) || fail "imatrix sha differs after copy"
cp $CC/manifest.json $ARC/calib_manifest.json
echo "IMATRIX_ARCHIVED $ARC $(grep ' imatrix.dat' $ARC/imatrix.sha256)"
[ -n "${UPF:-}" ] && fail "upload failed (imatrix done and archived)"
echo "AGENTIC_DRIVE_OK $(date -u +%FT%TZ)"
