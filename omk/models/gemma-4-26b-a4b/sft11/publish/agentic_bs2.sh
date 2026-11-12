#!/bin/bash
# agentic_bs2.sh export|upload|imatrix -- bm-889 sft11-final publish, bs2 side (owner 2026-11-12 17:59 CET: "upload the weights to
#   HF in a private repo ManniX-ITA/gemma-4-26B-A4B-agentic / use bs2 gpu0 to create the full AtomicChat i-matrix").
#   export : omk_qad export of runs/g4_bf16_sft11/stage1/final.pt (step 2108/2108) to a NEW HF dir (CPU). The gate export
#            dir had its shards trimmed; this is the same deterministic export (same code, same .pt) re-run. Checks: 295
#            replaced / 718 unchanged-verified, replaced_keys == the gate export's list, every index shard present.
#   upload : create the PRIVATE repo (refuses if it exists), upload the HF dir without the two local provenance files
#            (LFS_SHA256.json is the BASE model's shard table, omk_bf16_export.json holds local paths), then hub LFS sha256
#            == local sha256 for every shard.
#   imatrix: canonical omnimergekit quantize_gguf.py (solidpc 54ea808 snapshot) --imatrix-only, GPU0 only: F16 base GGUF from
#            the HF dir, AtomicChat calib_train (official-template render, sha 33fef70c1f8d5965, re-rendered byte-identical
#            to the v9-agentic pod build), all chunks, --parse-special (script default), checkpoint every 1000 chunks.
#            Same flags as the v9-agentic AC imatrix (pod_v9_agentic_publish.sh stage imatrix). imatrix.dat + F16 KEPT.
# Sentinels AGENTIC_<STAGE>_OK / AGENTIC_<STAGE>_FAIL. Never overwrites a dir, a repo, a GGUF or an imatrix.
set -uo pipefail
ST=${1:?usage: agentic_bs2.sh export|upload|imatrix}
G=/mnt/sdc/omk-model-optimizer/g4_recovery
CK=$G/runs/g4_bf16_sft11/stage1/final.pt
M=/mnt/sdc/ml/models/gemma-4-26B-A4B-it
HF=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
OLD=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal/omk_bf16_export.json
GG=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic-GGUF
REPO=ManniX-ITA/gemma-4-26B-A4B-agentic
CAL=/mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt; CSHA=33fef70c1f8d5965
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
OMK=/mnt/sdc/ml/omk_snap_54ea808/scripts
LC=/mnt/sdc/ml/llama.cpp-b9700
PYV=/mnt/sdc/omk-model-optimizer/venv/bin/python
PY=/srv/ml/envs/envs/omnimergekit/bin/python
U=$(echo $ST | tr a-z A-Z)
say(){ echo ">>> $(date -u +%FT%TZ) $*"; }
fail(){ echo "AGENTIC_${U}_FAIL $(date -u +%FT%TZ) $*"; exit 1; }
sdcfree(){ df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9; }

case $ST in
export)
  [ -e $HF ] && fail "$HF exists (never overwritten)"
  [ -e $G/export_agentic.log ] && fail "$G/export_agentic.log exists"
  [ -s $CK ] && [ -s ${CK%.pt}.json ] || fail "checkpoint or sidecar missing"
  f=$(sdcfree); [ "$f" -ge 120 ] || fail "/mnt/sdc ${f}G free, need 120 (48G HF + 50G F16 + margin)"
  say "export $CK -> $HF (CPU) / sdc ${f}G free"
  cd $G && PYTHONPATH=$G/omk_sft2_code CUDA_VISIBLE_DEVICES="" nice -n 10 $PYV -m omk_qad.train --phase export --mode bf16 \
    --model $M --load $CK --out-hf $HF > $G/export_agentic.log 2>&1 < /dev/null || fail "export: $(tail -2 $G/export_agentic.log | tr '\n' ' ' | cut -c1-300)"
  $PY - $HF $OLD <<'P' || fail "export checks"
import json, os, sys
hf, old = sys.argv[1:]
e = json.load(open(f"{hf}/omk_bf16_export.json")); o = json.load(open(old))
assert e["replaced"] == 295 and e["unchanged_verified"] == 718, (e["replaced"], e["unchanged_verified"])
assert e["step"] == 2108 and e["checkpoint"].endswith("stage1/final.pt"), e["checkpoint"]
assert sorted(e["replaced_keys"]) == sorted(o["replaced_keys"]), "replaced_keys differ from the gate export"
idx = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"]
sh = sorted(set(idx.values())); miss = [s for s in sh if not os.path.getsize(os.path.join(hf, s))]
assert not miss, miss
print("EXPORT_CHECK replaced 295 unchanged 718 keys==gate shards", len(sh), "tensors", len(idx))
P
  [ "$(sha256sum $HF/chat_template.jinja | cut -d' ' -f1)" = $TSHA ] || fail "chat_template is not the official ae53464b"
  ls -la $HF; echo "AGENTIC_EXPORT_OK $(date -u +%FT%TZ)";;
upload)
  grep -q '^AGENTIC_EXPORT_OK' $G/agentic_export.out 2>/dev/null || fail "no AGENTIC_EXPORT_OK"
  say "upload $HF -> $REPO (private)"
  $PY - $HF $REPO <<'P' || fail "upload"
import hashlib, json, os, sys
from huggingface_hub import HfApi
hf, repo = sys.argv[1:]; api = HfApi()
try:
    api.repo_info(repo); sys.exit(f"{repo} already exists -- refusing (never overwritten)")
except SystemExit: raise
except Exception: pass
api.create_repo(repo, private=True, repo_type="model")
assert api.repo_info(repo).private, "repo is not private"
api.upload_folder(folder_path=hf, repo_id=repo, ignore_patterns=["LFS_SHA256.json", "omk_bf16_export.json"],
                  commit_message="bm-889 sft11 final (step 2108/2108), BF16 safetensors")
info = api.model_info(repo, files_metadata=True)
hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
bad = []
for f in sorted(os.listdir(hf)):
    if f in ("LFS_SHA256.json", "omk_bf16_export.json"): continue
    if f not in hub: bad.append(f"missing {f}"); continue
    if f.endswith(".safetensors"):
        h = hashlib.sha256()
        with open(os.path.join(hf, f), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        if h.hexdigest() != hub[f]: bad.append(f"sha {f}")
        else: print("SHA_OK", f, h.hexdigest()[:16])
assert not bad, bad
print("REPO", repo, "private", api.repo_info(repo).private, "commit", info.sha, "files", len(hub))
P
  echo "AGENTIC_UPLOAD_OK $(date -u +%FT%TZ)";;
imatrix)
  grep -q '^AGENTIC_EXPORT_OK' $G/agentic_export.out 2>/dev/null || fail "no AGENTIC_EXPORT_OK"
  [ -e $GG ] && fail "$GG exists (never overwritten)"
  [ "$(sha256sum $CAL | cut -c1-16)" = $CSHA ] || fail "calib sha"
  [ "$(sha256sum $HF/chat_template.jinja | cut -d' ' -f1)" = $TSHA ] || fail "template sha"
  f=$(sdcfree); [ "$f" -ge 60 ] || fail "/mnt/sdc ${f}G free, need 60 for the F16"
  [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1000 ] || fail "GPU0 not empty"
  mkdir -p $GG; say "imatrix on GPU0 (AtomicChat full corpus, --parse-special, save every 1000) / sdc ${f}G free"
  export CUDA_VISIBLE_DEVICES=0 LLAMA_CPP_HOME=$LC
  $PY $OMK/quantize_gguf.py --model $HF --output-dir $GG --imatrix-only \
    --cal-data $CAL --imatrix-chunks -1 --imatrix-save-frequency 1000 --imatrix-timeout 172800 \
    --chat-template-file $HF/chat_template.jinja --chat-template-sha256 $TSHA \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local < /dev/null || fail "quantize_gguf --imatrix-only"
  [ -s $GG/imatrix.dat ] || fail "no imatrix.dat"
  ls -la $GG; sha256sum $GG/imatrix*.dat
  echo "AGENTIC_IMATRIX_OK $(date -u +%FT%TZ)";;
*) fail "unknown stage";;
esac
