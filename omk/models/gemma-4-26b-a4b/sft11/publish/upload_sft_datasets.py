#!/usr/bin/env python3
"""bm-889: upload the sft11 corpus datasets to PRIVATE HF dataset repos (owner 2026-11-12 21:25 CET: "add the datasets
used, if not present in HF to HF private repos on my account: like the kimi traces rendered by gemma4-31b").

For each staged dir: refuse if the repo exists, create it private, upload_folder, then every file must be on the hub and
every LFS file's sha256 must equal SHA256SUMS. Sentinel DS_UPLOAD_OK / DS_UPLOAD_FAIL. Never overwrites a repo.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

S = Path("/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/hf_datasets")
JOBS = [("ManniX-ITA/hermes-agent-kimi-traces-gemma4-31b", S / "hermes-agent-kimi-traces-gemma4-31b"),
        ("ManniX-ITA/gemma-4-26B-A4B-agentic-sft", S / "gemma-4-26B-A4B-agentic-sft")]


def main() -> None:
    api = HfApi()
    for repo, d in JOBS:
        sums = {ln.split("  ", 1)[1].removeprefix("./"): ln.split("  ", 1)[0]
                for ln in (d / "SHA256SUMS").read_text().splitlines()}
        local = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and not any(q.startswith(".") for q in p.relative_to(d).parts))
        missing_sum = [f for f in local if f not in sums and f not in ("SHA256SUMS", "README.md", "LICENSE.md")]
        if missing_sum:
            sys.exit(f"DS_UPLOAD_FAIL {repo}: files without a SHA256SUMS line {missing_sum}")
        if api.repo_exists(repo, repo_type="dataset"):
            sys.exit(f"DS_UPLOAD_FAIL {repo} already exists -- refusing (never overwritten)")
        api.create_repo(repo, repo_type="dataset", private=True)
        assert api.repo_info(repo, repo_type="dataset").private, "repo is not private"
        print(f"UPLOAD {repo} <- {d} ({len(local)} files)", flush=True)
        api.upload_folder(folder_path=str(d), repo_id=repo, repo_type="dataset",
                          commit_message="bm-889 sft11 corpus (Gemma-4-26B-A4B agentic SFT)",
                          ignore_patterns=[".*", ".*/**", "**/.*", "**/.*/**"])
        info = api.dataset_info(repo, files_metadata=True)
        hub = {s.rfilename: s for s in info.siblings}
        bad = [f"missing {f}" for f in local if f not in hub]
        n_lfs = 0
        for f, s in hub.items():
            if s.lfs is not None:
                n_lfs += 1
                if sums.get(f) != s.lfs.sha256:
                    bad.append(f"sha {f}")
        if bad:
            sys.exit(f"DS_UPLOAD_FAIL {repo}: {bad[:10]}")
        print(f"REPO {repo} private {api.repo_info(repo, repo_type='dataset').private} commit {info.sha} "
              f"files {len(hub)} lfs_sha_ok {n_lfs}", flush=True)
    print("DS_UPLOAD_OK")


if __name__ == "__main__":
    main()
