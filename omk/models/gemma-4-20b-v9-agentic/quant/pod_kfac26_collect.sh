#!/bin/bash
# pod_kfac26_collect.sh -- RUNS ON bs2. Pulls every finished 26B pod ladder tier (pod_kfac26_ladder.sh: .tier_done)
# into the bs2 ladder dir runs/g4_agentic_kfac/ladder/<TIER>/ where the HF publisher, the ollama pusher and the
# solidpc archiver pick it up unchanged. Never overwrites an existing bs2 kfac.gguf (a tier built on both hosts is
# reported, the bs2 one wins). Verifies the pod sidecar sha after the pull (pod->bs2 measured 9 MB/s: ~35 min for an
# 18 GB tier), appends the pod's LT_TWIN line to the bs2 twins.log, marks .collected on the pod and removes the pod's
# kfac.gguf (the pod copy is staging; bs2 + hub + solidpc keep the file). Sentinels PC_START / PC_TIER / PC_FAIL /
# PC_DONE in runs/pod_kfac26_collect.log.  usage: pod_kfac26_collect.sh   (HOURS=48)
set -u
KEY=/root/.ssh/id_ed25519_vastpod; H=${POD_HOST:-ssh1.vast.ai}; P=${POD_PORT:-29914}
S="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $P"
R=/mnt/sdc/omk-model-optimizer/runs; K=$R/g4_agentic_kfac; L=$R/pod_kfac26_collect.log; D=/workspace/kfac26
HOURS=${HOURS:-48}
say() { echo "[pc $(date -u +%FT%TZ)] $*" | tee -a $L; }
say "PC_START pod $H:$P:$D -> $K/ladder, $HOURS h"
end=$(( $(date +%s) + HOURS * 3600 ))
while [ $(date +%s) -lt $end ]; do
  for T in $($S root@$H "cd $D/ladder 2>/dev/null && for d in */; do d=\${d%/}; [ -e \$d/.tier_done ] && [ ! -e \$d/.collected ] && echo \$d; done" 2>/dev/null); do
    B=$K/ladder/$T; mkdir -p $B
    if [ -s $B/kfac.gguf ]; then say "PC_FAIL $T: bs2 already has a kfac.gguf, pod build left in place (not collected)"; continue; fi
    free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
    [ "$free" -ge 60 ] || { say "$T: /mnt/sdc ${free}G < 60G, waiting"; continue; }
    want=$($S root@$H "cut -d' ' -f1 $D/ladder/$T/kfac.gguf.sha256"); [ ${#want} = 64 ] || { say "PC_FAIL $T: no pod sha"; continue; }
    t0=$(date +%s)
    rsync -a --partial --inplace --exclude 'ptq.gguf' --exclude '*.partial_*' -e "$S" root@$H:$D/ladder/$T/ $B/ >> $L 2>&1 || { say "PC_FAIL $T: rsync (resumes next cycle)"; continue; }
    got=$(sha256sum $B/kfac.gguf | cut -d' ' -f1)
    [ "$got" = "$want" ] || { say "PC_FAIL $T: sha $got != pod $want (partial kept for resume)"; continue; }
    rm -f $B/.tier_done
    grep "LT_TWIN $T " $B/../twins.log >/dev/null 2>&1 || $S root@$H "grep 'LT_TWIN $T ' $D/ladder/twins.log" | sed "s/^\[pl /[lt 26b $T pod /" >> $K/ladder/twins.log
    $S root@$H "touch $D/ladder/$T/.collected && rm -f $D/ladder/$T/kfac.gguf"
    say "PC_TIER $T collected $(( $(stat -c %s $B/kfac.gguf) / 1048576 )) MiB in $(( $(date +%s) - t0 )) s, sha ${want:0:12} OK; pod copy removed"
  done
  sleep 300
done
say "PC_DONE $HOURS h elapsed"
