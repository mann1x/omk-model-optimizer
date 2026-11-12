#!/bin/bash
# bm-889 sft6 Phase 2 -> 3 -> 4 chain (owner 2026-11-06 18:22 CET "when it's done start immediately the training, even
# if it fails again"; no more C3 retries, topup_c3 stopped 17:23Z). On bs2, detached. Steps:
#  1 wait until no anti-rep run is queued or running (4 h cap)
#  2 final admission (admission_r3.py, report in phase2/)            -> ADMISSION_OK
#  3 admitted runs -> SDK capture dirs, eval split by run (p3_export) -> EXPORT_OK
#  4 C8 (gen_kimi31b_v1) and C3/C4 (gen_cpif26b_v1) gated_audit/kept.strict.jsonl -> conv (sft2_assemble.py: history
#    turns score:false, only the regenerated turn trained)                -> ASSEMBLE_OK x2
#  5 render 32k --pack, CPU, in parallel: rollouts --capture (4 train shards + eval) verified against the served
#    template (:8266 /apply-template, c13 eos106 GGUF = the rollout server); conv files --history-reasoning strip.
#    Gate: every stats complete, refused <= 2 % of requests, train windows >= 200; 0-window files removed.
#  6 overlap gate (sft2_overlap.py: no scored eval token ends a 200-gram seen in train) -> OVERLAP_OK
#  7 CORPUS_MANIFEST.json (sha256 + windows per file)
#  8 stop OUR rollout servers (r3_watch if alive, proxies, upstreams :8266 GPU1 / :8268 GPU0; cmdline-checked), GPU0 free
#  9 start sft6_train.sh detached; wait for SFT6_START and the first training step (30 min cap)
# Sentinels P34_OK / P34_FAIL <step> <why>. Never overwrites a build, a render or a run.
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery
S=$G/sft6_build; L=$S/loopproxy; P=$S/phase3
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
A=/mnt/sdc/automaticum/bin/automaticum
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it
export AUTOMATICUM_CONFIG=/mnt/sdc/automaticum/config
TAG=v2; OUT=$P/build_$TAG; R=$P/render_32k_$TAG; RREL=sft6_build/phase3/render_32k_$TAG
ts(){ date -u +%FT%TZ; }
say(){ echo ">>> $(ts) $*"; }
fail(){ echo "P34_FAIL $(ts) $*"; exit 1; }
gpu(){ nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
for d in $OUT $R $G/runs/g4_bf16_sft6 $G/train_sft6.log; do [ -e "$d" ] && fail "0 $d exists (never overwritten)"; done
for f in $P/admission_r3.py $P/p3_export.py $P/sft6_train.sh $G/sft2_build/sft2_assemble.py $G/sft2_build/sft2_overlap.py \
  $S/gen_kimi31b_v1/gated_audit/kept.strict.jsonl $S/gen_cpif26b_v1/gated_audit/kept.strict.jsonl $S/sft6_loop_proxy.py; do
  [ -s "$f" ] || fail "0 missing $f"; done

say "1 wait for the rollout queue"
end=$(( $(date +%s) + 4 * 3600 ))
while "$A" run ls -limit 500 2>&1 | grep antirep | awk '$2=="queued" || $2=="running"' | grep -q .; do
  [ "$(date +%s)" -gt "$end" ] && fail "1 queue not empty after 4 h"
  sleep 60
done
say "queue empty"

say "2 final admission"
"$PY" $P/admission_r3.py $S > $P/admission_final.log 2>&1
grep -q '^ADMISSION_OK' $P/admission_final.log || fail "2 admission: $(tail -3 $P/admission_final.log | tr '\n' ' ' | cut -c1-300)"
REP=$(grep '^report ' $P/admission_final.log | tail -1 | cut -d' ' -f2)
[ -s "$REP" ] || fail "2 no report path"
say "$(grep '^ADMISSION_OK' $P/admission_final.log) $REP"

say "3 export captures"
"$PY" $P/p3_export.py "$REP" $OUT > $P/export_$TAG.log 2>&1
grep -q '^EXPORT_OK' $P/export_$TAG.log || fail "3 export: $(tail -3 $P/export_$TAG.log | tr '\n' ' ' | cut -c1-300)"
say "$(tail -3 $P/export_$TAG.log | tr '\n' ' ')"

say "4 assemble C8 + C3/C4"
"$PY" $G/sft2_build/sft2_assemble.py $S/gen_kimi31b_v1/gated_audit/kept.strict.jsonl $OUT/conv_c8 --eval-min 32 > $P/assemble_c8.log 2>&1
grep -q '^ASSEMBLE_OK' $P/assemble_c8.log || fail "4 assemble c8: $(tail -2 $P/assemble_c8.log | tr '\n' ' ')"
"$PY" $G/sft2_build/sft2_assemble.py $S/gen_cpif26b_v1/gated_audit/kept.strict.jsonl $OUT/conv_cpif --eval-min 16 > $P/assemble_cpif.log 2>&1
grep -q '^ASSEMBLE_OK' $P/assemble_cpif.log || fail "4 assemble cpif: $(tail -2 $P/assemble_cpif.log | tr '\n' ' ')"
say "c8 $(grep '^ASSEMBLE_OK' $P/assemble_c8.log)  cpif $(grep '^ASSEMBLE_OK' $P/assemble_cpif.log)"

say "5 render"
mkdir -p $R/logs || fail "5 mkdir"
export PYTHONPATH=$G/omk_sft4_code CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
curl -sf -m 10 http://127.0.0.1:8266/props > /dev/null || fail "5 :8266 (served template for --verify-server) not answering"
i=0
for d in $OUT/capture_train/run-*; do mkdir -p $OUT/capture_train_s$((i % 4)); mv "$d" $OUT/capture_train_s$((i % 4))/; i=$((i + 1)); done
pids=""
for k in 0 1 2 3 eval; do
  if [ $k = eval ]; then c=$OUT/capture_eval; o=$R/rollouts.eval.roles.jsonl; else c=$OUT/capture_train_s$k; o=$R/rollouts_s$k.train.roles.jsonl; fi
  nice -n 10 "$PY" -m omk_qad.rollouts --model $M --capture $c --template-kwargs '{"enable_thinking": true}' \
    --verify-server http://127.0.0.1:8266 --seq 32768 --stride 16384 --pack --out $o > $R/logs/rollouts_$k.log 2>&1 &
  pids="$pids $!"
done
for f in $OUT/conv_c8/*.conv.jsonl $OUT/conv_cpif/*.conv.jsonl; do
  [ -s "$f" ] || continue
  n=$(basename "$f" .conv.jsonl)
  nice -n 10 "$PY" -m omk_qad.rollouts --model $M --in "$f" --history-reasoning strip --seq 32768 --pack \
    --out $R/$n.roles.jsonl > $R/logs/$n.log 2>&1 &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait "$p" || rc=1; done
[ $rc = 0 ] || fail "5 a render failed: $(grep -l -iE 'error|Traceback' $R/logs/*.log | head -3 | tr '\n' ' ')"
"$PY" - $R <<'PYG' || fail "5 render gate"
import glob, json, os, sys
R = sys.argv[1]
req = ref = tw = 0
for s in sorted(glob.glob(f"{R}/*.roles.stats.json")):
    st = json.load(open(s))
    assert st.get("complete") is True, f"{s} incomplete"
    name = os.path.basename(s).removesuffix(".roles.stats.json")
    if st["windows"] == 0:
        os.remove(s); os.remove(s.replace(".stats.json", ".jsonl")); print(f"removed empty {name}"); continue
    r = sum((st.get("refused") or {}).values())
    req += st.get("requests", 0); ref += r if st.get("requests") else 0
    tw += st["windows"] * name.endswith(".train")
    print(f"{name:34} windows {st['windows']:5} requests {st.get('requests', '-')} refused {r} cut {st.get('cut', '-')} "
          f"roles {st.get('tokens_by_role')}")
print(f"train windows {tw}; rollout requests {req}, refused {ref}")
assert tw >= 200, f"only {tw} train windows"
assert ref <= 0.02 * max(req, 1), f"refused {ref} of {req} rollout requests > 2 %"
print("RENDER_GATE_OK")
PYG
say "render gate ok"

say "6 overlap gate"
"$PY" $G/sft2_build/sft2_overlap.py $R > $R/logs/overlap.log 2>&1
grep -q '^OVERLAP_OK' $R/logs/overlap.log || fail "6 overlap: $(tail -6 $R/logs/overlap.log | tr '\n' ' ' | cut -c1-400)"
[ -e $R/OVERLAP_OK ] || fail "6 no OVERLAP_OK file"

say "7 manifest"
"$PY" - $R "$REP" $OUT <<'PYM' || fail "7 manifest"
import glob, hashlib, json, os, sys
R, rep, out = sys.argv[1:4]
files = {}
for f in sorted(glob.glob(f"{R}/*.roles.jsonl")):
    st = json.load(open(f.replace(".roles.jsonl", ".roles.stats.json")))
    files[os.path.basename(f)] = {"sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(), "windows": st["windows"],
                                  "tokens_by_role": st.get("tokens_by_role"), "requests": st.get("requests"),
                                  "refused": st.get("refused"), "cut": st.get("cut")}
m = {"name": "bm889-sft6-corpus", "seq": 32768,
     "chosen_by_owner": "2026-11-06 18:22 CET: start the training when the rollout queue is done",
     "sources": {"rollouts": f"admitted runs of {rep} (round-3 loop rules, PASS, net < task limit, audit clean), "
                             "one example per captured request, verified vs served /apply-template (:8266 c13 eos106)",
                 "c8": "gen_kimi31b_v1/gated_audit/kept.strict.jsonl (31B regen of kimi turns)",
                 "cpif": "gen_cpif26b_v1/gated_audit/kept.strict.jsonl (26B cp/if regen, checker-verified)"},
     "split": json.load(open(f"{out}/split.json")), "overlap_gate": "sft2_overlap.py OVERLAP_OK", "files": files}
json.dump(m, open(f"{R}/CORPUS_MANIFEST.json", "w"), indent=1)
print("MANIFEST", len(files), sum(v["windows"] for k, v in files.items() if ".train." in k), "train windows")
PYM

say "8 stop our rollout servers"
stop(){  # pidfile pattern
  local p; p=$(cat "$L/$1" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null || { echo "$1: not running"; return 0; }
  tr '\0' ' ' < /proc/$p/cmdline | grep -q -- "$2" || fail "8 $1 pid $p is not ours ($2)"
  kill "$p"; for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
  kill -0 "$p" 2>/dev/null && fail "8 $1 pid $p still alive after 120 s"
  echo "$1 pid $p stopped"
}
stop r3_watch.pid r3_watch.sh
stop proxy.pid "sft6_loop_proxy.py"
stop proxy_g0.pid "sft6_loop_proxy.py"
stop upstream.pid "--port 8266"
stop upstream_g0.pid "--port 8268"
for _ in $(seq 60); do [ "$(gpu 0)" -lt 2000 ] && break; sleep 2; done
[ "$(gpu 0)" -lt 2000 ] || fail "8 GPU0 still holds $(gpu 0) MiB"
say "GPU0 $(gpu 0) MiB, GPU1 $(gpu 1) MiB"

say "9 start sft6 training"
cd $G || fail "9 cd"
setsid nohup bash $P/sft6_train.sh $RREL > $G/sft6_train.out 2>&1 < /dev/null &
echo $! > $G/sft6_train.pid
end=$(( $(date +%s) + 1800 ))
until grep -qE '^SFT6_(START|FAIL)' $G/sft6_train.out 2>/dev/null; do
  [ "$(date +%s)" -gt "$end" ] && fail "9 no SFT6_START after 30 min: $(tail -3 $G/sft6_train.out | tr '\n' ' ')"
  sleep 10
done
grep -q '^SFT6_FAIL' $G/sft6_train.out && fail "9 $(grep '^SFT6_FAIL' $G/sft6_train.out)"
say "$(grep '^SFT6_START' $G/sft6_train.out)"
until grep -qE '"step": *1[,}]|^step +1 |step=1 |QAD_STEP 1 ' $G/train_sft6.log 2>/dev/null; do
  kill -0 "$(cat $G/sft6_train.pid)" 2>/dev/null || fail "9 launcher died: $(tail -3 $G/sft6_train.out | tr '\n' ' ')"
  [ "$(date +%s)" -gt $((end + 1800)) ] && { say "WARN no step-1 line matched in 30 min (pattern); trainer alive"; break; }
  sleep 20
done
echo "P34_OK $(ts) trainer launcher pid $(cat $G/sft6_train.pid)"
