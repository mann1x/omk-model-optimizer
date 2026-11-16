#!/usr/bin/env bash
# kfac_ladder_archive.sh — RUNS ON solidpc. Streams every finished K-FAC ladder tier off bs2 (24 MiB/s) into persistent
# storage, so the two ladders (≈125 G 20B + ≈165 G 26B of kfac.gguf) never have to fit on /mnt/sdc (168 G free, lanes
# wait below 70 G). Every 5 min, for both models and every ladder/<TIER>/ with kfac.gguf + kfac.eval.json and no
# .archived_solidpc marker: rsync (resumable) kfac.gguf + sidecars/logs/scores, verify sha256 (sidecar, or computed on
# bs2 when missing), then touch .archived_solidpc on bs2. Nothing on bs2 is deleted unless the owner's marker
# <K>/ladder/.rm_kfac_archived_granted exists — then the bs2 kfac.gguf of an ARCHIVED tier is removed (logged, rm -v).
# Archive: $ARCH/{26b_agentic,20b_v9agentic_sft12}/<TIER>/   Sentinels KA_START / KA_TIER / KA_RM / KA_FAIL / KA_DONE.
set -u
BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
ARCH=$BM/google/kfac_ladders
LOG=$ARCH/archive.log
HOURS=${HOURS:-48}
BS2="ssh -o ConnectTimeout=20 bs2"
RUNS=/mnt/sdc/omk-model-optimizer/runs
say() { echo "[ka $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
mkdir -p $ARCH; say "KA_START archive $ARCH, $HOURS h"
end=$(( $(date +%s) + HOURS * 3600 ))
while [ $(date +%s) -lt $end ]; do
  for spec in "26b_agentic $RUNS/g4_agentic_kfac" "20b_v9agentic_sft12 $RUNS/g4_20b_kfac"; do
    set -- $spec; M=$1; K=$2
    for T in $($BS2 "cd $K/ladder 2>/dev/null && for d in */; do d=\${d%/}; [ -s \$d/kfac.gguf ] && [ -s \$d/kfac.eval.json ] && [ ! -e \$d/.archived_solidpc ] && echo \$d; done" 2>/dev/null); do
      D=$K/ladder/$T; A=$ARCH/$M/$T; mkdir -p $A
      want=$($BS2 "cd $D && { [ -s kfac.gguf.sha256 ] || sha256sum kfac.gguf > kfac.gguf.sha256; }; cut -d' ' -f1 kfac.gguf.sha256")
      [ ${#want} = 64 ] || { say "KA_FAIL $M $T: no sha from bs2 ('$want')"; continue; }
      t0=$(date +%s)
      rsync -a --partial --inplace --include='kfac.gguf' --include='*.sha256' --include='*.json' --include='*.log' --exclude='*' \
            bs2:$D/ $A/ >> $LOG 2>&1 || { say "KA_FAIL $M $T: rsync"; continue; }
      got=$(sha256sum $A/kfac.gguf | cut -d' ' -f1)
      [ "$got" = "$want" ] || { say "KA_FAIL $M $T: sha $got != bs2 $want (partial copy kept for resume)"; continue; }
      sz=$(stat -c %s $A/kfac.gguf)
      $BS2 "touch $D/.archived_solidpc"
      say "KA_TIER $M $T archived $((sz / 1048576)) MiB in $(( $(date +%s) - t0 )) s, sha ${want:0:12} OK -> $A"
    done
    # owner-granted reclaim (2026-11-16 22:29 CET "bs2 archive-copy deletion granted") of bs2 copies that are archived
    # (sha-verified above) AND published to HF AND pushed to ollama (the bs2 pusher needs the file: .published_ollama)
    if $BS2 "test -e $K/ladder/.rm_kfac_archived_granted"; then
      for T in $($BS2 "cd $K/ladder && for d in */; do d=\${d%/}; [ -e \$d/.archived_solidpc ] && [ -e \$d/.published_hf ] && [ -e \$d/.published_ollama ] && [ -s \$d/kfac.gguf ] && echo \$d; done" 2>/dev/null); do
        [ "$(sha256sum $ARCH/$M/$T/kfac.gguf 2>/dev/null | cut -d' ' -f1)" = "$(cut -d' ' -f1 $ARCH/$M/$T/kfac.gguf.sha256)" ] \
          || { say "KA_FAIL $M $T: archive copy does not verify, NOT removing bs2 copy"; continue; }
        say "KA_RM $M $T: $($BS2 "rm -v $K/ladder/$T/kfac.gguf" 2>&1 | tr '\n' ' ') (archived copy verified)"
      done
    fi
  done
  sleep 300
done
say "KA_DONE $HOURS h elapsed"
