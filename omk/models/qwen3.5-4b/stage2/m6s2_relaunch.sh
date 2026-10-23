#!/usr/bin/env bash
# m6s2_relaunch.sh — bm-885 stage-2 RESTART launcher (BR, LR from the env; runs m6s2_stage2_br.sh), generated from
# m6s2_launch.sh; identical except the script, the per-branch status/pid and the claim reason. Original header:
# m6s2_launch.sh — bm-885 stage-2 launcher, runs on solidpc. Start it DETACHED (it outlives the Claude session):
#     setsid nohup bash omk_qad_work/m6/m6s2_launch.sh > omk_qad_work/m6/m6s2_launch.log 2>&1 < /dev/null &
# Claims bs2/gpu0 with `claude-hooks-res run` (waits up to 2h if someone holds it; the claim lives exactly as long as
# this wrapper, not the session; it is also what opencoti's resource-lock.sh sees), starts m6s2_stage2.sh detached on bs2 (a
# dropped ssh cannot kill the run), polls its status file, and releases the claim when it ends. ~4.6 h of GPU0.
# Sentinels: M6S2_LAUNCH_DONE / M6S2_LAUNCH_PAUSED / M6S2_LAUNCH_FAIL. Pause overnight: touch the PAUSE file (see
# m6s2_stage2.sh); relaunching this script resumes from the last checkpoint.
set -uo pipefail
BR="${BR:?BR must be set}"; LR="${LR:?LR must be set}"; export BR LR
B=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models; M=$B/omk_qad_work/m6
RES=/root/.local/bin/claude-hooks-res
M6S2=/mnt/sdc/omk-model-optimizer/runs/m6s2; ST=$M6S2/stage2_$BR.status
WHY="backup_models bm-885: omk_qad M6 iq2 stage-2 RESTART branch $BR lr $LR (2400 steps + export + KLD, ~5.5h)"
say(){ echo "[s2-launch $(date -u +%FT%TZ)] $*"; }

if [ "${1:-}" != --held ]; then
  scp -q "$M/m6s2_stage2_br.sh" "$M/m6s2_plan.py" bs2:$M6S2/ || { say "M6S2_LAUNCH_FAIL scp"; exit 1; }
  ssh bs2 "test -s $M6S2/plan.json" || { say "M6S2_LAUNCH_FAIL no plan.json on bs2 (run m6s2_plan.py)"; exit 1; }
  say "waiting for bs2/gpu0 (claude-hooks-res run, timeout 2h)"
  $RES run bs2/gpu0 -r "$WHY" --ttl 8h --timeout 2h -- bash "$0" --held
  rc=$?
  say "claim released, rc=$rc"
  exit $rc
fi

# ---- inside the claude-hooks-res claim. NO second resource-lock.sh claim: since 2026-10-22 it is a thin wrapper over
# claude-hooks-res, so a `wait bs2-gpu0` here queues behind this very job's claim and deadlocks (bug-831).
say "claim held; starting m6s2_stage2.sh detached on bs2"
ssh bs2 "rm -f $ST $M6S2/stage2_$BR.pid; BR=$BR LR=$LR setsid nohup bash $M6S2/m6s2_stage2_br.sh > /dev/null 2>&1 < /dev/null &" \
  || { say "M6S2_LAUNCH_FAIL ssh start"; exit 1; }
sleep 20   # let it write its pid and first status (a start delay, not a readiness test: the poll below decides)
fails=0 last=""
while :; do
  if out=$(ssh -o ConnectTimeout=20 bs2 "cat $ST 2>/dev/null; p=\$(cat $M6S2/stage2_$BR.pid 2>/dev/null); \
           [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null && echo ALIVE || echo GONE"); then
    fails=0
    s=$(echo "$out" | head -1); alive=$(echo "$out" | tail -1)
    [ "$s" != "$last" ] && say "bs2: $s" && last=$s
    case "$s" in
      S2_DONE*) say "M6S2_LAUNCH_DONE"; exit 0 ;;
      S2_PAUSED*) say "M6S2_LAUNCH_PAUSED (relaunch this script to resume)"; exit 0 ;;
      S2_FAIL*) say "M6S2_LAUNCH_FAIL $s (log: $M6S2/stage2_$BR.log)"; exit 1 ;;
    esac
    [ "$alive" = GONE ] && { say "M6S2_LAUNCH_FAIL stage2 process gone without a final status: $s"; exit 1; }
  else
    fails=$((fails + 1))
    say "ssh poll failed ($fails in a row); the bs2 run is detached and continues"
    [ $fails -ge 180 ] && { say "M6S2_LAUNCH_FAIL bs2 unreachable for 3h; claims released, run state unknown"; exit 1; }
  fi
  sleep 60
done
