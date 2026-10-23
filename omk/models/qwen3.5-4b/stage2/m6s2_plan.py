"""bm-885 stage-2 cache plan: weights for mix_v2 such that the cache never restarts (repeats) a source.

    python m6s2_plan.py ROOT N_SEQ SHARD SEED OUT.json      # CPU only; ROOT = /mnt/sdc/omk-model-optimizer
The cache (omk_qad.train.phase_cache) draws each window's source with random.Random(SEED).choices per SHARD and
packs every source with _packed, which silently restarts a source that runs out (only cache.json source_passes
records it). Here: windows per source counted with the trainer's own _texts exactly as _packed packs them (ids +
eos, seq-token windows, remainder dropped), weights = those capacities, and the cache's draws replayed with the same
seed to prove demand <= capacity for every source. Writes the --data string and the table; prints PLAN_OK/PLAN_FAIL.
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

from omk_qad.train import _texts

R, N_SEQ, SHARD, SEED, OUT = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
SEQ = 1024  # the stage-1 cache's window (cache.json seq)
D2 = R / "data" / "mix_v2"
files = sorted(D2.glob("*.train.jsonl")) + sorted((D2 / "fresh").glob("*.fresh.jsonl"))
tok = AutoTokenizer.from_pretrained(R / "models" / "Qwen3.5-4B")
cap = {}
for p in files:
    n = 0
    for ids, add_eos in _texts(str(p), tok):
        n += len(ids) + (1 if add_eos and tok.eos_token_id is not None else 0)
    cap[str(p)] = n // SEQ
    print(f"{p.relative_to(D2)}: {n} tokens -> {cap[str(p)]} windows", flush=True)

srcs = list(cap)
total = sum(cap.values())
w = [round(cap[s] / total, 6) for s in srcs]  # the exact floats the cache parses back from the --data string
rng, demand = random.Random(SEED), [0] * len(srcs)
for s0 in range(0, N_SEQ, SHARD):
    for k in rng.choices(range(len(srcs)), w, k=min(SHARD, N_SEQ - s0)):
        demand[k] += 1
rows = [{"file": s, "windows": cap[s], "weight": w[k], "demand": demand[k],
         "headroom": cap[s] - demand[k]} for k, s in enumerate(srcs)]
print(f"{'source':48} {'windows':>8} {'weight':>8} {'demand':>7} {'headroom':>8}")
for r in rows:
    print(f"{Path(r['file']).name:48} {r['windows']:8d} {r['weight']:8.4f} {r['demand']:7d} {r['headroom']:8d}")
print(f"total windows {total} ({total * SEQ / 1e6:.2f}M tokens); n_seq {N_SEQ} = {N_SEQ / total:.1%} of them")
ok = all(r["headroom"] >= 0 for r in rows)
OUT.write_text(json.dumps({"n_seq": N_SEQ, "shard": SHARD, "seed": SEED, "seq": SEQ, "rows": rows, "ok": ok,
                           "data": ",".join(f"{r['file']}={r['weight']}" for r in rows)}, indent=1))
print("PLAN_OK" if ok else "PLAN_FAIL")
sys.exit(0 if ok else 1)
