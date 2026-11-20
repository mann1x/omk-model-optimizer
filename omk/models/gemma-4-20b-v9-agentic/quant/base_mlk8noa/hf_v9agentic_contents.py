#!/usr/bin/env python3
"""What is in each private v9-agentic repo, folder by folder (owner 2026-11-20 06:19: "what's in this repo?").
Read-only. GiB per top-level folder (files at the root are grouped by extension); every LFS file >= 1 GiB listed.
"""

from collections import defaultdict

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

api = HfApi()
REPOS = [
    "ManniX-ITA/omk-qad-artifacts-gemma4-agentic",
    "ManniX-ITA/gemma-4-20B-v9-agentic",
    "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF",
    "ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11",
    "ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-GGUF",
]
for rid in REPOS:
    by = defaultdict(lambda: [0, 0])
    big = []
    for e in api.list_repo_tree(rid, recursive=True, expand=True):
        if not isinstance(e, RepoFile):
            continue
        sz = e.lfs.size if e.lfs else (e.size or 0)
        parts = e.path.split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) == 2 else "ROOT " + e.path.rsplit(".", 1)[-1])
        by[key][0] += 1
        by[key][1] += sz
        if sz >= 2**30:
            big.append((sz, e.path))
    print(f"\n=== {rid}  total {sum(v[1] for v in by.values())/2**30:.1f} GiB")
    for k, (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {s/2**30:8.2f} GiB  {n:4d} files  {k}")
    for sz, p in sorted(big, reverse=True):
        print(f"      {sz/2**30:6.2f}  {p}")
