#!/bin/bash
# run_summary.sh ID VERDICT WALL_MS -- on bs2: save one finished gate run's events (logs --events -json) under
#   gate_analysis_loop/sft11_s400_live/, run loopscan.py + timeline.py (unchanged, from gate_analysis_loop/) and print
#   run_summary.py's line. Never overwrites: an existing events file is reused, not rewritten. Reads automaticum only.
set -uo pipefail
ID=$1; V=$2; W=$3; STEP=${4:-400}
D=/mnt/sdc/omk-model-optimizer/g4_recovery/gate_analysis_loop; O=$D/sft11_s${STEP}_live
mkdir -p $O
cd /mnt/sdc/automaticum/repo || { echo "$ID SUMMARY_FAIL no repo"; exit 1; }
export AUTOMATICUM_CONFIG=/mnt/sdc/automaticum/config
E=$O/$ID.events.jsonl
if [ ! -s $E ]; then
  timeout 300 ../bin/automaticum logs "$ID" --events -json > $E.tmp 2>/dev/null && mv $E.tmp $E || { echo "$ID SUMMARY_FAIL logs"; exit 1; }
fi
[ -s $O/$ID.loopscan.json ] || nice -n 10 python3 $D/loopscan.py "$ID" < $E > $O/$ID.loopscan.json
[ -s $O/$ID.timeline.json ] || nice -n 10 python3 $D/timeline.py "$ID" < $E > $O/$ID.timeline.json
python3 $O/../sft11_run_summary.py "$ID" "$V" "$W" $O/$ID.loopscan.json $O/$ID.timeline.json $E || echo "$ID SUMMARY_FAIL summarise"
