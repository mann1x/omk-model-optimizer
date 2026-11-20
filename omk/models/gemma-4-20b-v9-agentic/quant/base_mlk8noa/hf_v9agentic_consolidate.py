#!/usr/bin/env python3
"""Consolidate the 20B v9-agentic Hub repos (owner 2026-11-20 06:54 CET: "do 1-4 fix").

The base-mlk8-noa-sft11 build replaces the abandoned sft11+sft12 build inside the model's own repos.
Everything here is server-side (rename repo, in-repo LFS copy): no bytes are re-uploaded.

  1. rename the old repos aside:   gemma-4-20B-v9-agentic[-GGUF]            -> ...-sft12-OLD
  2. move the new repos onto the real names: ...-base-mlk8-noa-sft11[-GGUF] -> gemma-4-20B-v9-agentic[-GGUF]
  3. in the GGUF repo rename every <long>-<TIER>.gguf to gemma-4-20B-v9-agentic-<TIER>.gguf (CommitOperationCopy +
     Delete, one commit), rewrite the .gguf.sha256 files (they carry the filename) and the README (names, links,
     an "internal build, not released" line); the weights repo gets no file renames (shards are generic)
  4. delete the two -sft12-OLD repos, then super_squash_history both live repos
  5. verify: file list + LFS sha256 of every GGUF in the new repo == what the old names had

Each step is idempotent; a step that finds its work done says so and moves on. Sentinels CONS_<STEP> / CONS_FAIL.
Token: the hf CLI login. Run on bs2 omnimergekit py3.11.  --dry prints the plan only.
"""

import sys
import time

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

OWNER = "ManniX-ITA"
NAME = "gemma-4-20B-v9-agentic"
LONG = "gemma-4-20B-v9-agentic-base-mlk8-noa-sft11"
OLD_SUFFIX = "-sft12-OLD"
DRY = "--dry" in sys.argv
api = HfApi()
log = open("/mnt/sdc/omk-model-optimizer/runs/hf_v9agentic_consolidate.log", "a")


def say(*a):
    msg = f"[cons {time.strftime('%FT%TZ', time.gmtime())}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def exists(rid):
    try:
        api.repo_info(rid)
        return True
    except RepositoryNotFoundError:
        return False


def files(rid):
    return {e.path: e for e in api.list_repo_tree(rid, recursive=True, expand=True) if isinstance(e, RepoFile)}


def die(*a):
    say("CONS_FAIL", *a)
    sys.exit(1)


say("start", "DRY" if DRY else "LIVE", "account", api.whoami()["name"])

# ---------- snapshot of the new GGUF repo before anything moves (for the step-5 check)
src_gguf = f"{OWNER}/{LONG}-GGUF" if exists(f"{OWNER}/{LONG}-GGUF") else f"{OWNER}/{NAME}-GGUF"
before = {p: (e.lfs.sha256 if e.lfs else None) for p, e in files(src_gguf).items() if p.endswith(".gguf")}
if not before:
    die("no .gguf in", src_gguf)
say("snapshot", src_gguf, len(before), "gguf")
if any(not p.startswith(LONG) and not p.startswith(NAME) for p in before):
    die("unexpected gguf names", [p for p in before if not p.startswith(LONG)][:3])

# ---------- 1. rename old aside, 2. move new onto the real names
for suffix in ("", "-GGUF"):
    real, long_, old = f"{OWNER}/{NAME}{suffix}", f"{OWNER}/{LONG}{suffix}", f"{OWNER}/{NAME}{OLD_SUFFIX}{suffix}"
    if exists(long_):
        if exists(real):
            if exists(old):
                die("both", real, "and", old, "exist; refusing to guess")
            say("1. rename", real, "->", old)
            if not DRY:
                api.move_repo(from_id=real, to_id=old)
        say("2. move", long_, "->", real)
        if not DRY:
            api.move_repo(from_id=long_, to_id=real)
    else:
        say("1-2. done already:", real, "is the live repo;", old, "exists" if exists(old) else "absent")
say("CONS_MOVED")

# ---------- 3. rename the GGUFs + sha256 files, rewrite the README
gg = f"{OWNER}/{NAME}-GGUF"
cur = files(gg) if not DRY else files(src_gguf)
ops = []
for p, e in sorted(cur.items()):
    if not p.startswith(LONG + "-"):
        continue
    new = NAME + p[len(LONG):]
    if p.endswith(".gguf"):
        ops += [CommitOperationCopy(src_path_in_repo=p, path_in_repo=new), CommitOperationDelete(path_in_repo=p)]
    elif p.endswith(".sha256"):
        txt = open(hf_hub_download(src_gguf if DRY else gg, p)).read()
        txt = txt.replace(LONG, NAME)
        ops += [CommitOperationAdd(path_in_repo=new, path_or_fileobj=txt.encode()), CommitOperationDelete(path_in_repo=p)]
    else:
        die("unexpected long-named file", p)
    say("3. rename", p, "->", new)
if "README.md" in cur:
    md = open(hf_hub_download(src_gguf if DRY else gg, "README.md")).read()
    md2 = md.replace(LONG, NAME)
    md2 = md2.replace(f"# {NAME} — GGUF (K-FAC quantizations)",
                      f"# {NAME} — GGUF (K-FAC quantizations)\n\n> **Internal build `base-mlk8-noa-sft11`, under "
                      "evaluation — not released.** It replaces the earlier sft11+sft12 build of this model in this "
                      "repo; the owner is not satisfied with its stability (loops / non-convergence under "
                      "anti-repetition sampling) and a further SFT round is planned.", 1)
    if md2 == md:
        say("3. README needs no change")
    else:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=md2.encode()))
        say("3. README rewritten", len(md), "->", len(md2), "chars")
if ops:
    if not DRY:
        api.create_commit(gg, operations=ops, commit_message=f"{NAME}: files renamed to the repo name (build base-mlk8-noa-sft11); card updated")
    say("3. commit", len(ops), "operations")
else:
    say("3. nothing to rename")
say("CONS_RENAMED")

# ---------- 4. delete the OLD repos, squash the live ones
for suffix in ("", "-GGUF"):
    old = f"{OWNER}/{NAME}{OLD_SUFFIX}{suffix}"
    if exists(old):
        say("4. delete", old)
        if not DRY:
            api.delete_repo(old)
    else:
        say("4.", old, "absent")
for suffix in ("", "-GGUF"):
    live = f"{OWNER}/{NAME}{suffix}"
    if DRY:
        continue
    n = len(api.list_repo_commits(live))
    if n > 1:
        api.super_squash_history(live, commit_message=f"{NAME}{suffix}: build base-mlk8-noa-sft11 (history squashed 2026-11-20)")
        say("4. squashed", live, n, "->", len(api.list_repo_commits(live)))
    else:
        say("4.", live, "already 1 commit")
say("CONS_SQUASHED")

# ---------- 5. verify
if not DRY:
    after = {p: (e.lfs.sha256 if e.lfs else None) for p, e in files(gg).items() if p.endswith(".gguf")}
    want = {NAME + p[len(LONG):] if p.startswith(LONG) else p: s for p, s in before.items()}
    if after != want:
        die("gguf set differs", sorted(set(after) ^ set(want))[:5],
            [p for p in want if p in after and after[p] != want[p]][:5])
    leftovers = [p for p in files(gg) if LONG in p]
    if leftovers:
        die("long names left", leftovers[:5])
    w = files(f"{OWNER}/{NAME}")
    say("5. verified", len(after), "gguf sha-identical under the new names;", len(w), "files in the weights repo")
    say("CONS_DONE", f"https://huggingface.co/{OWNER}/{NAME}", f"https://huggingface.co/{OWNER}/{NAME}-GGUF")
else:
    say("CONS_DRY_DONE")
