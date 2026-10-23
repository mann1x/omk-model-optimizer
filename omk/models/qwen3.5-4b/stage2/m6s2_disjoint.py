"""bm-885 gate: no mix_v2 train document is a stage-1 train or eval document (content hash), and the token budgets.

    python m6s2_disjoint.py ROOT     # ROOT = /mnt/sdc/omk-model-optimizer
Stage-1 side: mix_v1 train+eval files, corpus_v1 train prefixes the cache read (fresh manifest docs_skipped) and the
corpus_v1 eval files. A document is its rendered text ({"text"}) or its row JSON (messages rows), normalised.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

R = Path(sys.argv[1])
D1, D2 = R / "data" / "mix_v1", R / "data" / "mix_v2"


def key(row) -> str:
    """What the trainer reads of a row: its messages, else its text, else (other schemas) the whole row."""
    if "messages" in row:
        body = json.dumps(row["messages"], sort_keys=True)
    elif "text" in row:
        body = row["text"]
    else:
        body = json.dumps(row, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def rows(p: Path, limit=None):
    """Parsed JSONL rows of ``p`` (from its first ``limit`` lines only, when given)."""
    with p.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            if line.strip():
                yield json.loads(line)


seen = {}
for p in sorted(D1.glob("*.jsonl")):
    seen.update((key(r), f"mix_v1/{p.name}") for r in rows(p))
for p in sorted((R / "data" / "corpus_v1").glob("*eval*.jsonl")):
    seen.update((key(r), f"corpus_v1/{p.name}") for r in rows(p))
fresh = json.loads((D2 / "fresh" / "manifest.json").read_text())["sources"]
for src, m in fresh.items():
    p = Path(src)
    if p.suffix == ".jsonl":  # the prefix the stage-1 cache read
        seen.update((key(r), f"stage1-prefix/{p.name}") for r in rows(p, m["docs_skipped"]))
    else:
        it = (r for b in pq.ParquetFile(p).iter_batches(batch_size=256, columns=["text"]) for r in b.to_pylist())
        for i, r in enumerate(it):
            if i >= m["docs_skipped"]:
                break
            seen[key(r)] = f"stage1-prefix/{p.name}"
print(f"stage-1 side: {len(seen)} distinct documents")

bad, total = 0, 0
train = sorted(D2.glob("*.train.jsonl")) + sorted((D2 / "fresh").glob("*.fresh.jsonl"))
man2 = json.loads((D2 / "manifest.json").read_text())["sources"]
for p in train:
    n = hits = 0
    for r in rows(p):
        n += 1
        if key(r) in seen:
            hits += 1
    total += n
    bad += hits
    tok = (man2.get(p.name.split(".")[0], {}).get("train", {}).get("tokens")
           or next((m["tokens"] for m in fresh.values() if m["file"] == p.name), None))
    print(f"{p.relative_to(D2)}: {n} docs, {tok} tokens, {hits} reused")
toks = sum(m["train"]["tokens"] for m in man2.values()) + sum(m["tokens"] for m in fresh.values())
print(f"mix_v2 train: {total} docs, {toks / 1e6:.2f}M tokens, {bad} reused")
print("DISJOINT_OK" if bad == 0 else "DISJOINT_FAIL")
