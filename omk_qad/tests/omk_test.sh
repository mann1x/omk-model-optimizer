#!/usr/bin/env bash
# omk_test.sh -- the omk_qad test runner: named, resumable, pausable runs; suites; impact selection; profile.
#
#   omk_test.sh run NAME [pytest args...]   start a detached run (one per $OMK_QAD_WORK; refuses while another is alive)
#                                           e.g.  run q8 --suite q8        run imp --impact        run full
#                                                 run one omk_qad/tests/test_q8_native.py::test_x
#   omk_test.sh resume NAME                 relaunch the same argv; tests already passed in NAME.jsonl are skipped
#   omk_test.sh pause NAME | cont NAME      SIGSTOP / SIGCONT the whole process group (train subprocesses included)
#   omk_test.sh stop NAME                   SIGINT pytest: the current test ends, the summary is written, resume later
#   omk_test.sh status [NAME]               alive?, counts from NAME.jsonl, last test, log tail
#   omk_test.sh impact [REV|--files P...]   which tests a change reaches (omk_qad/tests/impact.py)
#   omk_test.sh suites                      the suite table (pytest --list-suites)
#   omk_test.sh profile [N] [NAME]          N slowest tests + per-file totals from durations.jsonl (or NAME.jsonl)
#
# Files: $OMK_QAD_WORK/runs/NAME.{argv,pid,log,jsonl}   (OMK_QAD_WORK default: the backup_models omk_qad_work/pytest dir)
# Env:   OMK_PY (default omk-modelopt python)  OMK_TEST_GPU=1 to leave CUDA visible (default: hidden = the CPU suite)
# The suite needs `-o addopts=""` (upstream's addopts want plugins the env lacks) -- added here, never repeat it.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
PY=${OMK_PY:-/root/anaconda3/envs/omk-modelopt/bin/python}
WORK=${OMK_QAD_WORK:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest}
RUNS=$WORK/runs; mkdir -p "$RUNS"
CMD=${1:-help}; shift || true
die() { echo "omk_test: $*" >&2; exit 1; }
alive() { [ -s "$RUNS/$1.pid" ] && kill -0 "$(cat "$RUNS/$1.pid")" 2>/dev/null; }
any_alive() { for p in "$RUNS"/*.pid; do [ -e "$p" ] || continue; n=$(basename "$p" .pid); alive "$n" && echo "$n"; done; }
pgid_of() { ps -o pgid= -p "$(cat "$RUNS/$1.pid")" | tr -d ' '; }

launch() {  # NAME ; argv in $RUNS/NAME.argv (one arg per line)
  local n=$1 busy; busy=$(any_alive)
  [ -n "$busy" ] && die "run '$busy' is alive in $WORK -- one pytest session per OMK_QAD_WORK (bug-968); stop or wait"
  mapfile -t ARGS < "$RUNS/$n.argv"
  local gpu=""; [ "${OMK_TEST_GPU:-0}" = 1 ] || gpu="CUDA_VISIBLE_DEVICES="
  cd "$REPO" || die "cd $REPO"
  env ${gpu:+CUDA_VISIBLE_DEVICES=} OMK_QAD_WORK="$WORK" OMK_PYTEST_FAILLOG="$RUNS/$n.failures.log" \
    setsid nohup "$PY" -m pytest -o addopts="" -p no:cacheprovider -q --durations=25 \
    --resume "$RUNS/$n.jsonl" "${ARGS[@]}" >> "$RUNS/$n.log" 2>&1 < /dev/null &
  echo $! > "$RUNS/$n.pid"
  echo "omk_test: run '$n' pid $! (pgid $!) log $RUNS/$n.log progress $RUNS/$n.jsonl"
  echo "          argv: ${ARGS[*]}"
}

case $CMD in
  run)
    n=${1:?NAME}; shift
    [ -e "$RUNS/$n.argv" ] && ! alive "$n" && die "run '$n' exists -- 'resume $n' continues it, or pick a new name"
    [ $# -eq 0 ] && set -- omk_qad/tests
    printf '%s\n' "$@" > "$RUNS/$n.argv"
    echo "# $(date -Is) run $n" >> "$RUNS/$n.log"
    launch "$n" ;;
  resume)
    n=${1:?NAME}; [ -s "$RUNS/$n.argv" ] || die "no run '$n'"
    alive "$n" && die "run '$n' is still alive (pid $(cat "$RUNS/$n.pid")); 'cont' if paused, 'stop' first otherwise"
    echo "# $(date -Is) resume $n" >> "$RUNS/$n.log"
    launch "$n" ;;
  pause|cont)
    n=${1:?NAME}; alive "$n" || die "run '$n' is not alive"
    sig=STOP; [ "$CMD" = cont ] && sig=CONT
    kill -"$sig" -- -"$(pgid_of "$n")" && echo "omk_test: $CMD '$n' (SIG$sig to pgid $(pgid_of "$n"))" ;;
  stop)
    n=${1:?NAME}; alive "$n" || die "run '$n' is not alive"
    kill -CONT -- -"$(pgid_of "$n")" 2>/dev/null; kill -INT "$(cat "$RUNS/$n.pid")" \
      && echo "omk_test: stop '$n' (SIGINT; the current test finishes, then the summary; 'resume $n' later)" ;;
  status)
    for a in "$RUNS"/*.argv; do [ -e "$a" ] || continue; n=$(basename "$a" .argv); [ -n "${1:-}" ] && [ "$n" != "$1" ] && continue
      st=ended; alive "$n" && { st=alive; grep -q "T (stopped)" <(ps -o stat= -p "$(cat "$RUNS/$n.pid")" | sed 's/^T.*/T (stopped)/') && st=PAUSED; }
      "$PY" - "$RUNS/$n.jsonl" "$n" "$st" <<'PY'
import json, sys
from pathlib import Path
p, name, st = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
last = {}; sessions = 0; secs = 0.0; latest = "-"
for ln in (p.read_text().splitlines() if p.exists() else []):
    try: r = json.loads(ln)
    except ValueError: continue
    if "session_start" in r: sessions += 1
    if r.get("when") == "call" or (r.get("when") == "setup" and r.get("outcome") == "skipped"):
        last[r["nodeid"]] = r["outcome"]; secs += r.get("duration", 0); latest = r["nodeid"]
c = {o: sum(1 for v in last.values() if v == o) for o in ("passed", "failed", "skipped")}
print(f"{name:12s} {st:7s} sessions {sessions}  passed {c['passed']} failed {c['failed']} skipped {c['skipped']}  "
      f"test-time {secs/60:.0f} min  last: {latest}")
PY
      grep -h "OMK_SELECT\|passed\|failed" "$RUNS/$n.log" 2>/dev/null | tail -n 2 | cut -c1-200 | sed 's/^/             /'
    done ;;
  impact) cd "$REPO" && exec "$PY" -m omk_qad.tests.impact "$@" ;;
  suites) cd "$REPO" && exec env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -o addopts="" -p no:cacheprovider omk_qad/tests --list-suites 2>/dev/null ;;
  profile)
    N=${1:-25}; src=$WORK/durations.jsonl; [ -n "${2:-}" ] && src=$RUNS/$2.jsonl
    "$PY" - "$src" "$N" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path
p, n = Path(sys.argv[1]), int(sys.argv[2])
last = {}
for ln in (p.read_text().splitlines() if p.exists() else []):
    try: r = json.loads(ln)
    except ValueError: continue
    if r.get("when") == "call": last[r["nodeid"]] = (r["duration"], r["outcome"])
if not last: sys.exit(f"no call records in {p}")
per_file = defaultdict(lambda: [0.0, 0])
for nid, (d, _) in last.items():
    f = nid.split("::")[0].split("/")[-1]; per_file[f][0] += d; per_file[f][1] += 1
tot = sum(v[0] for v in per_file.values())
print(f"{p}: {len(last)} tests, {tot/60:.1f} min of test time (call phase only)\n")
print("per file (min, tests, share):")
for f, (d, c) in sorted(per_file.items(), key=lambda kv: -kv[1][0]):
    print(f"  {d/60:7.1f}  {c:4d}  {100*d/tot:5.1f}%  {f}")
print(f"\n{n} slowest:")
for nid, (d, o) in sorted(last.items(), key=lambda kv: -kv[1][0])[:n]:
    print(f"  {d:7.1f}s {o:7s} {nid}")
PY
    ;;
  help|*) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
