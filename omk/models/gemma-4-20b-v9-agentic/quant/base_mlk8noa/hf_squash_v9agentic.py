#!/usr/bin/env python3
"""Squash the history of the owner's PRIVATE v9-agentic Hub repos (owner 2026-11-20 06:01 CET:
"squash the HF private repos of v9-agentic to recover Hub storage quota").

super_squash_history folds every commit of main into one; LFS objects referenced only by the
dropped commits stop counting toward private storage. Public repos and repos with a single
commit are left alone. Prints commits before/after per repo. Token: the hf CLI login.
"""

import sys

from huggingface_hub import HfApi

REPOS = [
    "ManniX-ITA/gemma-4-20B-v9-agentic",
    "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF",
    "ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11",
    "ManniX-ITA/gemma-4-20B-v9-agentic-base-mlk8-noa-sft11-GGUF",
    "ManniX-ITA/omk-qad-artifacts-gemma4-agentic",
]
DRY = "--dry" in sys.argv
api = HfApi()
print("account", api.whoami()["name"], "DRY" if DRY else "LIVE")

for rid in REPOS:
    info = api.repo_info(rid)
    if not info.private:
        print(f"SKIP public {rid}")
        continue
    before = len(api.list_repo_commits(rid))
    if before <= 1:
        print(f"SKIP {rid} commits={before} (nothing to squash)")
        continue
    if DRY:
        print(f"WOULD squash {rid} commits={before}")
        continue
    api.super_squash_history(rid, commit_message=f"{rid.split('/')[-1]}: history squashed 2026-11-20 (storage reclaim)")
    after = len(api.list_repo_commits(rid))
    print(f"SQUASHED {rid} commits {before} -> {after}")

# reference only: the 26B weights repo holding the bm-892 final.pt parts
r26 = api.repo_info("ManniX-ITA/gemma-4-26B-A4B-agentic")
print("REF gemma-4-26B-A4B-agentic private=", r26.private, "commits=", len(api.list_repo_commits(r26.id)))
