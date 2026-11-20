#!/usr/bin/env python3
"""Inventory of the owner's PRIVATE v9-agentic Hub repos before a history squash.

For every private model/dataset under the account whose id mentions v9-agentic
(plus the artifacts repo, listed for reference only): commit count, number of
files, LFS bytes in the current tree. Also locates final.pt copies in the
artifacts repo (so the bs2 re-fetch can be deleted against a Hub copy).
Read-only. Token comes from the authenticated hf CLI login, never from argv.
"""

from huggingface_hub import HfApi

ARTIFACTS = "ManniX-ITA/omk-qad-artifacts-gemma4-agentic"
api = HfApi()
me = api.whoami()["name"]
print("account", me)


def size_of(repo_id, repo_type):
    tree = api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True, expand=True)
    n, lfs, plain = 0, 0, 0
    paths = []
    for e in tree:
        if e.__class__.__name__ != "RepoFile":
            continue
        n += 1
        paths.append(e.path)
        if e.lfs:
            lfs += e.lfs.size
        else:
            plain += e.size or 0
    return n, lfs, plain, paths


rows = []
for repo_type, lister in (("model", api.list_models), ("dataset", api.list_datasets)):
    for r in lister(author=me, limit=None):
        rid = r.id
        if "v9-agentic" not in rid.lower() and rid != ARTIFACTS:
            continue
        info = api.repo_info(rid, repo_type=repo_type)
        if not info.private:
            print(f"SKIP public  {repo_type:7s} {rid}")
            continue
        commits = api.list_repo_commits(rid, repo_type=repo_type)
        n, lfs, plain, paths = size_of(rid, repo_type)
        rows.append((repo_type, rid, len(commits), n, lfs))
        print(f"PRIVATE {repo_type:7s} {rid}  commits={len(commits)} files={n} lfs={lfs/2**30:.2f}GiB plain={plain/2**20:.1f}MiB")
        if rid == ARTIFACTS:
            for p in paths:
                if "final.pt" in p or "qadE6" in p:
                    print("   artifacts:", p)

print("TOTAL private v9-agentic lfs GiB:", sum(r[4] for r in rows) / 2**30)
