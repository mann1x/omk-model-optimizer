#!/usr/bin/env python3
"""hf_rm_previous_build_20b.py -- RUNS ON bs2 (omnimergekit env). Owner 2026-11-16 22:28 CET: "remove from 20b HF the
previous-build files". One commit deletes every .gguf / .gguf.sha256 of ManniX-ITA/gemma-4-20B-v9-agentic-GGUF that is
not (a) a file this ladder published (its hub LFS sha equals a .published_hf sha), (b) the sft12 F16 + sidecar, (c) the
mmproj (vision tower, valid for both builds), (d) imatrix.dat, README.md, .gitattributes. Same-named tiers of the old
build that the K-FAC ladder has not replaced yet go too (they are the wrong model; the publisher re-uploads the K-FAC
file when it lands). Prints the plan, then commits. Usage: hf_rm_previous_build_20b.py [--dry-run]
"""
import sys
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

REPO = "ManniX-ITA/gemma-4-20B-v9-agentic-GGUF"
LADDER = Path("/mnt/sdc/omk-model-optimizer/runs/g4_20b_kfac/ladder")
KEEP = {"README.md", ".gitattributes", "imatrix.dat", "imatrix.dat.sha256",
        "mmproj-gemma-4-20B-v9-agentic-F16.gguf", "gemma-4-20B-v9-agentic-F16.gguf", "gemma-4-20B-v9-agentic-F16.gguf.sha256"}


def main() -> None:
    dry = "--dry-run" in sys.argv
    api = HfApi()
    ours = {}
    for d in LADDER.iterdir():
        m = d / ".published_hf"
        if d.is_dir() and m.exists():
            ours[f"gemma-4-20B-v9-agentic-{d.name}.gguf"] = m.read_text().split()[0]
    for extra in (LADDER / ".published_extra_gemma-4-20B-v9-agentic-F16.gguf",):
        if extra.exists():
            ours["gemma-4-20B-v9-agentic-F16.gguf"] = extra.read_text().split()[0]
    info = api.model_info(REPO, files_metadata=True)
    files = {}
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        files[s.rfilename] = (getattr(s, "size", None), sha)
    delete, keep = [], []
    for name, (size, sha) in sorted(files.items()):
        base = name[:-7] if name.endswith(".sha256") else name
        if name in KEEP or name.startswith("calibration/"):
            keep.append((name, "fixed"))
        elif base in ours and (files.get(base, (None, None))[1] == ours[base]):
            keep.append((name, "K-FAC ladder, hub sha verified"))
        elif name.endswith(".gguf") or name.endswith(".gguf.sha256"):
            delete.append((name, size or 0))
        else:
            keep.append((name, "not a gguf"))
    tot = sum(s for _, s in delete)
    print(f"keep {len(keep)}:")
    for n, why in keep:
        print(f"  {n}  ({why})")
    print(f"delete {len(delete)} files, {tot / 1e9:.1f} GB:")
    for n, s in delete:
        print(f"  {n}  {s / 1e9:.2f} GB")
    if dry or not delete:
        print("DRY RUN, nothing deleted" if dry else "nothing to delete")
        return
    ops = [CommitOperationDelete(path_in_repo=n) for n, _ in delete]
    c = api.create_commit(repo_id=REPO, repo_type="model", operations=ops,
                          commit_message=f"remove the 2026-10-17 build's {len(delete)} files ({tot / 1e9:.0f} GB): this repo now holds the sft12 K-FAC ladder")
    print("HFRM_DONE", c.commit_url if hasattr(c, "commit_url") else c)


if __name__ == "__main__":
    main()
