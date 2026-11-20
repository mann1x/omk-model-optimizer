#!/usr/bin/env python3
"""Hub-side facts for the two bs2 deletions the owner authorised on 2026-11-20 06:01 CET.

  F16     ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-GGUF  <name>-F16.gguf   (LFS sha256 + size)
  final.pt ManniX-ITA/gemma-4-26B-A4B-agentic  qad/iq2_xxs_qadE6_kfacseed/stage1/final.pt.part00/.part01 + SHA256SUMS

Prints the values; the shell side compares them with the local sha256sum. Read-only.
"""

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
G = "ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-GGUF"
W = "ManniX-ITA/gemma-4-26B-A4B-agentic"
P = "qad/iq2_xxs_qadE6_kfacseed/stage1"

for e in api.list_repo_tree(G, expand=True):
    if e.path.endswith("-F16.gguf"):
        print(f"HUB_F16 {e.path} size={e.lfs.size} sha256={e.lfs.sha256}")

parts = {}
for e in api.list_repo_tree(W, path_in_repo=P, expand=True):
    name = e.path.rsplit("/", 1)[-1]
    if name.startswith("final.pt"):
        parts[name] = e
        print(f"HUB_PART {e.path} size={e.lfs.size if e.lfs else e.size} sha256={e.lfs.sha256 if e.lfs else '-'}")
print("HUB_PARTS_TOTAL", sum(e.lfs.size for n, e in parts.items() if n.startswith("final.pt.part")))
if "final.pt.SHA256SUMS" in parts:
    p = hf_hub_download(W, f"{P}/final.pt.SHA256SUMS")
    print("HUB_SHA256SUMS")
    print(open(p).read())
