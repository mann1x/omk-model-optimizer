"""bm-893: publish the 20B v9-agentic sft12 bf16 weights to the private HF repo, squash its history, verify, then
(after the K-FAC collector that read them has finished) remove the local copy from bs2 /mnt/sdc (owner 2026-11-16 13:24:
"publish the weights to the private HF repo for v9-agentic then squash it ... when uploaded, remove them from sdc").

Steps (each logged with a PUB_ sentinel; stops at the first failure, nothing is removed before the remote verifies):
  1. SHA256SUMS of every file to publish (written next to the weights)
  2. README: the current card with a build-replacement notice on top (the card's numbers are the previous build's)
  3. upload_folder: every file of the export dir; stale remote shards / index removed (delete_patterns)
  4. super_squash_history -> one commit holding exactly the new tree
  5. verify remote file list + sizes against the local dir
  6. wait for KFAC20_DONE (the 20B factor collector loaded this dir), then remove the local weights dir
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ManniX-ITA/gemma-4-20B-v9-agentic"
M = Path("/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12")
KFAC_LOG = Path("/mnt/sdc/omk-model-optimizer/runs/g4_20b_kfac/collect_g10.log")
SKIP = {"BUILD_OK", "SHA256SUMS"}
NOTICE = """> **Build replacement notice (2026-11-16).** The weights in this repository were replaced by the
> **sft12** build: the v9-agentic 98-expert prune rebuilt on the sft11 recovery base
> (`mlk8-noa`, refA recipe) plus the sft12 round (step 855; 180 tensors updated, 833 verified unchanged --
> see `omk_bf16_export.json`). The benchmark numbers, chart and loop comparison below were measured on the
> **previous** build and are kept for reference until the new quants (K-FAC vs llama.cpp imatrix, under
> evaluation) and their evals are published.

"""


def say(*a):
    print(f"[pub {time.strftime('%FT%TZ', time.gmtime())}]", *a, flush=True)


def fail(msg):
    say("PUB_FAIL", msg)
    sys.exit(1)


api = HfApi()
me = api.whoami()["name"]
say("PUB_START", REPO, "as", me, "from", M)
files = sorted(p for p in M.iterdir() if p.is_file() and p.name not in SKIP)
if not any(p.suffix == ".safetensors" for p in files) or not (M / "model.safetensors.index.json").exists():
    fail("no safetensors shards / index in the export dir")
# 1. checksums
lines = []
for p in files:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    lines.append(f"{h.hexdigest()}  {p.name}")
(M / "SHA256SUMS").write_text("\n".join(lines) + "\n")
say("PUB_SHA", len(lines), "files hashed")
# 2. README with the notice
readme = Path(api.hf_hub_download(REPO, "README.md", cache_dir=str(M.parent / ".hf_readme_cache"))).read_text()
if "Build replacement notice (2026-11-16)" not in readme:
    head, sep, body = readme.partition("\n---\n")  # YAML front matter ends at the second ---
    if not head.startswith("---") or not sep:
        fail("README front matter not found; not rewriting the card blindly")
    readme = head + sep + "\n" + NOTICE + body.lstrip("\n")
(M.parent / "README.sft12.md").write_text(readme)
# 3. upload
info = api.model_info(REPO, files_metadata=True)
if not info.private:
    fail("repo is not private; stopping (the owner asked for the private repo)")
before = {s.rfilename: s.size for s in info.siblings}
say("remote before:", len(before), "files,", f"{sum(v or 0 for v in before.values()) / 2**30:.1f}G")
api.upload_file(path_or_fileobj=str(M.parent / "README.sft12.md"), path_in_repo="README.md", repo_id=REPO,
                commit_message="Card: build replacement notice (sft12 weights)")
api.upload_folder(folder_path=str(M), repo_id=REPO, ignore_patterns=["BUILD_OK", "*.lock", ".hf_readme_cache/*"],
                  delete_patterns=["*.safetensors", "model.safetensors.index.json", "SHA256SUMS"],
                  commit_message="Weights: v9-agentic sft12 build (sft11 mlk8-noa base + sft12 step 855), 9 shards")
say("PUB_UPLOADED")
# 4. squash
api.super_squash_history(REPO, commit_message="v9-agentic sft12 weights (history squashed)")
commits = api.list_repo_commits(REPO)
say("PUB_SQUASHED commits now", len(commits))
# 5. verify
info = api.model_info(REPO, files_metadata=True)
remote = {s.rfilename: s.size for s in info.siblings}
want = {p.name: p.stat().st_size for p in files} | {"SHA256SUMS": (M / "SHA256SUMS").stat().st_size}
bad = [n for n, sz in want.items() if remote.get(n) != sz]
stale = [n for n in remote if n.endswith(".safetensors") and n not in want]
if bad or stale:
    fail(f"verify: size mismatch {bad} stale shards {stale}")
say("PUB_VERIFIED", len(want), "files match;", f"{sum(want.values()) / 2**30:.1f}G;",
    "remote extras:", sorted(set(remote) - set(want)))
json.dump({"repo": REPO, "files": want, "commits": len(commits), "squashed": True, "verified": time.time()},
          open(M.parent / "publish_sft12.json", "w"), indent=1)
# 6. remove the local weights once the collector that read them is done
while not (KFAC_LOG.exists() and any(t in KFAC_LOG.read_text() for t in ("KFAC20_DONE", "KFAC20_FAIL"))):
    time.sleep(60)
say("collector finished; removing local weights", M)
subprocess.run(["rm", "-rv", str(M)], check=True, stdout=subprocess.DEVNULL)
say("PUB_DONE", REPO, "local weights removed;", subprocess.run(["df", "--output=avail", "-BG", "/mnt/sdc"],
                                                              capture_output=True, text=True).stdout.split()[-1])
