#!/bin/bash
# watch_runs.sh -- bm-889 sft11 s400 gate: print one summary line per newly finished run (sft11_run_summary.sh on bs2),
#   remembering reported ids in watch_runs.seen so a re-armed monitor continues. Exits with ALL_DONE when 10/10 are
#   reported; prints WATCH_ERR on 5 failed polls in a row.
H=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/scripts/bm889_sft6/sft7/v11
STEP=${1:-400}
S=$H/watch_runs_s$STEP.seen; [ "$STEP" = 400 ] && S=$H/watch_runs.seen; touch $S
A=${2:-g4-128e-bf16-c14r15-bm889-sft11-s$STEP-gate}  # 2026-11-12: optional arm name (base / final gates), STEP = label
B=${3:-}  # 2026-11-12: optional batch id prefix (8244 gate: the first batch of the same arm was SKIPPED)
T=${4:-10}  # 2026-11-12: runs in the batch (h2h .277: 20)
D=/mnt/sdc/omk-model-optimizer/g4_recovery/gate_analysis_loop
bad=0
while :; do
  j=$(ssh -o ConnectTimeout=30 bs2 "cd /mnt/sdc/automaticum/repo; AUTOMATICUM_CONFIG=/mnt/sdc/automaticum/config ../bin/automaticum run ls -json -limit 60 -arm-name $A" < /dev/null 2>/dev/null)
  rows=$(echo "$j" | python3 -c '
import json, sys
r = json.load(sys.stdin)
for x in sorted(r, key=lambda x: x["id"]):
    if x.get("verdict") and x["id"].startswith(sys.argv[1]): print(x["id"], x["verdict"], x.get("wall_ms") or 0)
print("N", len(r))' "$B" 2>/dev/null)
  if [ -z "$rows" ]; then bad=$((bad+1)); [ $bad -ge 5 ] && { echo "WATCH_ERR run ls failed 5 times in a row"; bad=0; }; sleep 60; continue; fi
  bad=0
  echo "$rows" | grep -v '^N ' | while read -r id v w; do
    grep -qx "$id" $S && continue
    o=$(ssh -o ConnectTimeout=30 bs2 "bash $D/sft11_run_summary.sh $id $v $w $STEP" < /dev/null 2>&1 | tail -1)
    echo "$o"; echo "$id" >> $S
  done
  n=$(grep -c "^$B" $S)
  [ "$n" -ge "$T" ] && { echo "ALL_DONE $n/$T reported"; exit 0; }
  sleep 60
done
