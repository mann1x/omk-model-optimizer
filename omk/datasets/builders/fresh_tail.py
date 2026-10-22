#!/usr/bin/env python3
r"""The unread remainder of omk_qad cache sources, for a stage-2 corpus with no reused document (bm-885).

omk_qad's cache reads every source as one stream from its first document (``omk_qad.train._packed``), so a source
file listed again would repeat what an earlier cache trained on. This replays the trainer's own reader
(``omk_qad.train._texts``, same tokenizer) up to the windows that cache consumed, per its ``cache.json``
``rows_per_source``, and writes the documents AFTER the last one it touched, until ``--budget`` tokens:

    python fresh_tail.py --cache RUN/cache --tokenizer MODEL --out DIR \\
        --take corpus_v1/train_code.jsonl=2.0e6,fineweb/sample_10BT_000_00000.parquet=4.0e6,...

``--take`` paths are matched against the end of the cache's source paths. Rows are written unchanged (jsonl), or
as ``{"text": ...}`` (parquet), so a new cache tokenizes them exactly as the first one did. ``DIR/manifest.json``
records, per source: windows consumed, documents skipped, documents and tokens written, sha256.
"""
import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path):
    """The raw rows of one source, in the order omk_qad.train._texts reads them."""
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = [c for c in ("text", "messages") if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=256, columns=cols):
            yield from b.to_pylist()
    else:
        raise SystemExit(f"{path}: only .jsonl / .parquet have a remainder (a .txt is one document)")


def main():
    """Write each requested source's unread remainder and its manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="the earlier omk_qad cache dir (its cache.json)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--take", required=True, help="comma list of source-suffix=tokens")
    ap.add_argument("--margin", type=float, default=1.05, help="write budget * margin tokens (packer never restarts)")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from omk_qad.train import _texts

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    meta = json.loads((a.cache / "cache.json").read_text())
    seq, used = meta["seq"], meta["rows_per_source"]
    a.out.mkdir(parents=True, exist_ok=True)
    man = {"cache": str(a.cache), "tokenizer": a.tokenizer, "seq": seq, "margin": a.margin, "sources": {}}
    for kv in a.take.split(","):
        suffix, budget = kv.rsplit("=", 1)
        src = [p for p in used if p.endswith(suffix)]
        if len(src) != 1:
            raise SystemExit(f"{suffix}: matches {len(src)} cache sources")
        src = src[0]
        consumed = used[src] * seq  # tokens of this source the cache's windows hold
        target = int(float(budget) * a.margin)
        eos = tok.eos_token_id
        out = a.out / (Path(src).stem + ".fresh.jsonl")
        read = skipped = docs = toks = 0
        with out.open("w") as o:
            for row, (ids, add_eos) in zip(rows(Path(src)), _texts(src, tok)):
                n = len(ids) + (1 if add_eos and eos is not None else 0)
                if read < consumed:  # this document fed (all or part of) a cached window: never reused
                    read += n
                    skipped += 1
                    continue
                o.write(json.dumps(row if Path(src).suffix == ".jsonl" else {"text": row["text"]}) + "\n")
                docs += 1
                toks += n
                if toks >= target:
                    break
        if toks < target:
            raise SystemExit(f"{src}: only {toks} fresh tokens < {target}")
        man["sources"][src] = {"windows_consumed": used[src], "tokens_consumed": consumed, "docs_skipped": skipped,
                               "file": out.name, "docs": docs, "tokens": toks, "budget": int(float(budget)),
                               "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"{Path(src).name}: skipped {skipped} docs ({read} tok >= {consumed} consumed), wrote {docs} docs "
              f"{toks} tok -> {out.name}", flush=True)
        (a.out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("FRESH_DONE", flush=True)


if __name__ == "__main__":
    main()
