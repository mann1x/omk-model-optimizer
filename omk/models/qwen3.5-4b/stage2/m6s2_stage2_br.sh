#!/usr/bin/env bash
# m6s2_stage2_br.sh — bm-885 stage-2 RESTART variant (BR, LR from the env), generated from m6s2_stage2.sh. Owner
# 2026-10-23 00:27: "you have my go and my authorization to restart stage [2] with a lower LR if needed".
# Identical to m6s2_stage2.sh except: its own branch dir / GGUF / KLD row / status / log / pid, and --lr $LR.
# Original header:
# m6s2_stage2.sh — bm-885 stage 2 of the M6 iq2 student, the part that runs ON bs2 GPU0 (the claims are held on
# solidpc by m6s2_launch.sh, which starts this detached and polls $ST). Owner 2026-10-22: "yes write the launcher,
# we'll start it tonight".
#   1. cache: teacher top-64 over mix_v2 (21.02M fresh tokens, DISJOINT_OK vs stage 1), weights from m6s2_plan.py
#      (plan.json: no source is ever restarted), NO new imatrix (--imatrix-seqs 0), stage-1 eval files (same windows)
#   2. gates: CACHE_DONE, source_passes == [], rows per source == the plan's replay, eval tokens == stage-1 eval.pt
#   3. train: stage 2 branch mix_v2 from iq2/stage1/final.pt, stage-1 hyperparameters, stage-1 imatrix; W&B on by key
#   4. export the stage-2 GGUF (imatrix taken from the checkpoint), 5. KLD vs the same F16 reference as stage 1
# Re-runnable: a finished cache is gated not rebuilt, a stopped train resumes from last.pt, done steps are skipped.
# Stop cleanly overnight: touch $S2/PAUSE (checkpoints, exits; $ST = S2_PAUSED; relaunch m6s2_launch.sh to resume).
set -uo pipefail
BR="${BR:?BR (stage-2 branch) must be set}"; LR="${LR:?LR must be set}"
R=/mnt/sdc/omk-model-optimizer; RUN=$R/runs/m6_qwen35_4b; M6S2=$R/runs/m6s2
CODE=$R/code/omk-model-optimizer; PY=/root/anaconda3/envs/omk-modelopt/bin/python
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
C2=$RUN/cache_s2_mix_v2; S2=$RUN/iq2/stage2/$BR; GGUF=$S2/iq2_s2_$BR.gguf
PLAN=$M6S2/plan.json; ST=$M6S2/stage2_$BR.status; LOG=$M6S2/stage2_$BR.log
exec >> "$LOG" 2>&1
echo $$ > "$M6S2/stage2_$BR.pid"
say(){ echo "[s2 $(date -u +%FT%TZ)] $*"; }
status(){ echo "$1 $(date -u +%FT%TZ) ${2:-}" > "$ST"; say "STATUS $1 ${2:-}"; }
die(){ status S2_FAIL "$*"; exit 1; }
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$CODE   # GPU0 only: GPU1 is opencoti's
status RUNNING "preflight"

# ------------------------------------------------------------------------------------------------- preflight
grep -q "class Telemetry" "$CODE/omk_qad/train.py" || die "bs2 checkout has no W&B support"
$PY -c "import sys, wandb; sys.exit(wandb.__version__ != '0.26.1')" || die "omk-modelopt env lacks wandb==0.26.1"
{ [ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc; } || say "WARNING no W&B key on bs2: training runs without W&B"
[ -f "$RUN/iq2/stage1/final.pt" ] || die "no iq2 stage-1 final.pt"
[ -f "$RUN/cache/imatrix.pt" ] || die "no stage-1 imatrix.pt"
[ -f "$GGUF" ] && [ -f "$S2/final.pt" ] && [ -f "$S2/kld.done" ] && { status S2_DONE "already complete"; exit 0; }
$PY -c "import json,sys; p=json.load(open('$PLAN')); sys.exit(not (p['ok'] and p['n_seq']==19200 and p['seed']==0))" \
  || die "plan.json missing or not PLAN_OK for n_seq 19200 seed 0 (run m6s2_plan.py)"
free=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9); [ "$free" -ge 60 ] || die "/mnt/sdc has ${free}G free (< 60G)"
apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
[ -z "$apps" ] || die "GPU0 is not idle: $apps"
say "preflight OK: /mnt/sdc ${free}G free, GPU0 idle, plan OK"
if [ -n "${S2_DRY:-}" ]; then   # dry check: preflight only, no GPU work, no status left behind
  rm -f "$ST" "$M6S2/stage2_$BR.pid"; say "S2_DRY_OK"; exit 0
fi

# ----------------------------------------------------------------------------------------------------- cache
if [ ! -f "$C2/cache.json" ]; then
  status RUNNING "cache"
  DATA=$($PY -c "import json; print(json.load(open('$PLAN'))['data'])")
  EVAL=$($PY -c "import json; print(json.load(open('$RUN/cache/cache.json'))['args']['eval_data'])")
  cd "$CODE" || die "cd $CODE"
  $PY -W ignore::FutureWarning -m omk_qad.train --phase cache --model $R/models/Qwen3.5-4B --layout $RUN/iq2/layout.pt \
    --cache "$C2" --device cuda:0 --llama-cpp /opt/llama.cpp --seed 0 --data "$DATA" --eval-data "$EVAL" \
    --n-seq 19200 --n-eval 16 --seq 1024 --shard 1000 --topk 64 --imatrix-seqs 0 || die "cache phase rc $?"
fi
grep -q CACHE_DONE "$C2/cache.log" || die "cache.log has no CACHE_DONE"
$PY - "$C2" "$RUN/cache" "$PLAN" <<'EOF' || die "cache gates"
import json, os, sys
import torch
c2, c1, plan = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
m = json.load(open(f"{c2}/cache.json"))
assert m["source_passes"] == [], f"a source was restarted (repeated): {m['source_passes']}"
want = {r["file"]: r["demand"] for r in plan["rows"]}
assert m["rows_per_source"] == want, f"rows per source differ from the plan replay: {m['rows_per_source']} vs {want}"
assert not any(p.startswith("imatrix") for p in os.listdir(c2)), "the stage-2 cache made an imatrix"
e1, e2 = torch.load(f"{c1}/eval.pt"), torch.load(f"{c2}/eval.pt")
assert e1["families"] == e2["families"] and torch.equal(e1["x"], e2["x"]), "eval windows differ from stage 1"
same_idx = torch.equal(e1["idx"], e2["idx"])
print(f"cache gates OK: {m['n_seq']} seqs, rows per source == plan, no restart; eval tokens == stage 1, "
      f"teacher top-k idx identical={same_idx}, teacher ppl {e1['teacher_ppl']:.4f} / {e2['teacher_ppl']:.4f}")
EOF

# ----------------------------------------------------------------------------------------------------- train
if [ ! -f "$S2/final.pt" ]; then
  status RUNNING "train"
  cd "$CODE" || die "cd $CODE"
  $PY -W ignore::FutureWarning -m omk_qad.train --phase train --model $R/models/Qwen3.5-4B \
    --layout $RUN/iq2/layout.pt --cache "$C2" --imatrix $RUN/cache/imatrix.pt --out $RUN/iq2 --device cuda:0 \
    --llama-cpp /opt/llama.cpp --stage 2 --branch $BR --init-from $RUN/iq2/stage1/final.pt \
    --lr $LR --loss jaccard --steps 2400 --bs 8 --warmup 50 --refresh 4 --ggml-refresh 4 \
    --eval-every 300 --save-every 100 --save-minutes 20 --log-every 25
  rc=$?
  grep -E "WANDB (on|off)" "$S2/log.txt" | tail -1
  if tail -5 "$S2/log.txt" | grep -q "PAUSED at step"; then   # a pause exits 0: never export a paused student
    status S2_PAUSED "$(grep "PAUSED at step" "$S2/log.txt" | tail -1 | sed 's/.*PAUSED/PAUSED/')"; exit 0
  fi
  [ $rc -eq 0 ] || die "train rc $rc (see $S2/log.txt)"
  grep -q "STAGE_DONE stage 2 branch $BR" "$S2/log.txt" || die "train ended without STAGE_DONE"
fi

# ---------------------------------------------------------------------------------------------------- export
if [ ! -f "$GGUF" ]; then
  status RUNNING "export"
  cd "$CODE" || die "cd $CODE"
  $PY -W ignore::FutureWarning -m omk_qad.train --phase export --model $R/models/Qwen3.5-4B --layout $RUN/iq2/layout.pt \
    --device cuda:0 --load "$S2/final.pt" --gguf "$GGUF" --llama-cpp /opt/llama.cpp > "$S2/export.out" 2>&1 \
    || { rm -f "$GGUF"; die "export rc $? (see $S2/export.out)"; }
fi
$PY - "${GGUF%.gguf}.export.json" <<'EOF' || die "export report"
import json, sys
r = json.load(open(sys.argv[1]))
print({k: r.get(k) for k in ("verified", "verify_failures", "type_mismatches", "with_imatrix", "size_bytes")})
assert not r.get("verify_failures") and not r.get("type_mismatches"), "export verification failed"
EOF

# ------------------------------------------------------------------------------------------------------- KLD
if [ ! -f "$S2/kld.done" ]; then
  status RUNNING "kld"
  $OMKPY - "$RUN/compare/kld" "$GGUF" "M6-iq2-s2-$BR" <<'EOF' > "$S2/kld.out" 2>&1 || die "KLD (see $S2/kld.out)"
import json, sys
from pathlib import Path
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
import gguf_kld
kd, g = Path(sys.argv[1]), Path(sys.argv[2])
meta_p = next(kd.glob("kld_base_*.meta.json"))
basis = json.loads(meta_p.read_text())
basis["base_file"] = str(meta_p.with_name(meta_p.name.replace(".meta.json", ".bin")))  # not with_suffix: .meta.bin
ppl = gguf_kld.find_perplexity({"quantize": "/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-quantize"}, build_if_missing=False)
row = gguf_kld.measure(ppl, g, sys.argv[3], basis, kd / "wikitext-2-raw/wiki.test.raw")
gguf_kld.append_row(kd, row)
print("KLD_ROW", sys.argv[3], g.stat().st_size, row["kld_mean"], row["same_top_p_pct"], row["ppl_q"])
EOF
  grep -q KLD_ROW "$S2/kld.out" || die "KLD produced no row (see $S2/kld.out)"
  grep KLD_ROW "$S2/kld.out" | tee "$S2/kld.done"
fi

# ----------------------------------------------------------------------------------------------- comparison
$PY - "$RUN/iq2/stage1/summary.json" "$S2/summary.json" <<'EOF'
import json, sys
for tag, p in (("stage1", sys.argv[1]), ("stage2", sys.argv[2])):
    r = json.load(open(p))["results"]
    a, b = r[0], r[-1]
    print(f"{tag}: klk {a['klk']:.4f} -> {b['klk']:.4f}  top1 {a.get('top1', float('nan')):.4f} -> "
          f"{b.get('top1', float('nan')):.4f}  per family " +
          ", ".join(f"{f} {m['klk']:.4f}" for f, m in (b.get("families") or {}).items()))
EOF
$PY - "$RUN/compare/kld/kld_results.jsonl" "M6-iq2-s2-$BR" <<'EOF'
import json, sys
rows = {r["quant"]: r for r in map(json.loads, open(sys.argv[1])) if r.get("quant") in ("M6-iq2", "M6-iq2-s2-mix_v2", sys.argv[2])}
for q, r in rows.items():  # same F16 reference, wikitext-2 c512 x 100 chunks
    print(f"{q:18} {r['bytes'] / 1e9:.3f} GB  KLD {r['kld_mean'][0]:.4f} +- {r['kld_mean'][1]:.4f}  "
          f"same-top {r['same_top_p_pct'][0]:.2f}%  ppl {r['ppl_q'][0]:.3f}")
EOF
status S2_DONE "$(cat "$S2/kld.done")"
