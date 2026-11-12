#!/bin/bash
# agentic_quants_bs2.sh -- bm-889 sft11-final GGUF publish, bs2 (owner 2026-11-12 21:06 CET: "upload the fp16 bin to HF in a private
#   repo ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF with the i-matrix, create and upload Q4_K_M, Q6_K, Q8_0 quants" + 21:07 "use [the
#   AC i-matrix] for all quants, including q8").
#   Quants from the F16 (never from a quant), llama-quantize of the SAME b9700 build that made the imatrix, --imatrix on EVERY tier
#   (Q8_0 included, owner). Q4_K_M on the root fs (/mnt/sdc cannot hold all three; root stays >= 200G), Q6_K + Q8_0 next to the F16.
#   Each quant: KV check (quantize.imatrix.* present, eos 106, chat template 18683 B, general.file_type) + sha256 sidecar.
#   Upload: create the PRIVATE repo (refuses if it exists), F16 + imatrix.dat + 3 quants + sha256 sidecars, then hub LFS sha256 ==
#   local for every GGUF and the imatrix. Nothing is deleted. Sentinels AGQ_OK / AGQ_FAIL.
set -uo pipefail
GG=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic-GGUF
GR=/srv/ml/bm889-gguf/gemma-4-26B-A4B-agentic-GGUF
N=gemma-4-26B-A4B-agentic
F16=$GG/$N-F16.gguf; IM=$GG/imatrix.dat; IMSHA=609a0a041ddaa822da95d6d4abcb618d23c4ebf078b886275c7ad04ef9ff9974
Q=/mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-quantize
PY=/srv/ml/envs/envs/omnimergekit/bin/python
REPO=ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF
say(){ echo ">>> $(date -u +%FT%TZ) $*"; }
fail(){ echo "AGQ_FAIL $(date -u +%FT%TZ) $*"; exit 1; }
free(){ df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }

[ -s $F16 ] && [ -s $IM ] || fail "F16 or imatrix missing"
[ "$(sha256sum $IM | cut -d' ' -f1)" = $IMSHA ] || fail "imatrix sha differs from the archived one"
export CUDA_VISIBLE_DEVICES=""
# tier dir min_free_after_G
for spec in "Q4_K_M $GR 200" "Q6_K $GG 15" "Q8_0 $GG 5"; do
  set -- $spec; T=$1; D=$2; MIN=$3; OUT=$D/$N-$T.gguf
  mkdir -p $D
  [ -e $OUT ] && fail "$OUT exists (never overwritten)"
  need=$(( $(free $D) - $MIN )); est=$(case $T in Q4_K_M) echo 18;; Q6_K) echo 22;; Q8_0) echo 28;; esac)
  [ $need -ge $est ] || fail "$D: $(free $D)G free, $T needs ~${est}G and the floor is ${MIN}G"
  say "quantize $T -> $OUT (imatrix $IMSHA, free $(free $D)G)"
  nice -n 10 $Q --imatrix $IM $F16 $OUT $T > $D/quantize_$T.log 2>&1 < /dev/null || fail "llama-quantize $T: $(tail -3 $D/quantize_$T.log | tr '\n' ' ')"
  $PY - $OUT $T <<'P' || fail "KV check $T"
import sys
from gguf import GGUFReader
p, t = sys.argv[1:]; r = GGUFReader(p)
imk = sorted(k for k in r.fields if k.startswith("quantize.imatrix"))
assert imk, "no quantize.imatrix.* KV"
def v(k):
    f = r.fields[k]; return f.parts[f.data[0]]
eos = int(v("tokenizer.ggml.eos_token_id")[0]); tmpl = len(bytes(v("tokenizer.chat_template")))
ds = bytes(v("quantize.imatrix.dataset")).decode() if "quantize.imatrix.dataset" in r.fields else "?"
ch = int(v("quantize.imatrix.chunks_count")[0]) if "quantize.imatrix.chunks_count" in r.fields else -1
assert eos == 106, eos
assert tmpl == 18683, tmpl
print(f"KV_OK {t} imatrix_keys={len(imk)} dataset={ds} chunks={ch} eos={eos} template={tmpl}B file_type={int(v('general.file_type')[0])}")
P
  (cd $D && sha256sum $N-$T.gguf > $N-$T.gguf.sha256)
  say "$T done $(ls -l $OUT | awk '{print $5}') B sha $(cut -c1-16 $D/$N-$T.gguf.sha256) / $D free $(free $D)G"
done
[ -s $GG/$N-F16.gguf.sha256 ] || (cd $GG && sha256sum $N-F16.gguf > $N-F16.gguf.sha256)

say "upload -> $REPO (private)"
$PY - $REPO $GG $GR $N <<'P' || fail "upload"
import hashlib, os, sys
from huggingface_hub import HfApi
repo, gg, gr, n = sys.argv[1:]; api = HfApi()
try:
    api.repo_info(repo); sys.exit(f"{repo} already exists -- refusing (never overwritten)")
except SystemExit: raise
except Exception: pass
api.create_repo(repo, private=True, repo_type="model")
assert api.repo_info(repo).private
files = [f"{gg}/{n}-F16.gguf", f"{gg}/imatrix.dat", f"{gr}/{n}-Q4_K_M.gguf", f"{gg}/{n}-Q6_K.gguf", f"{gg}/{n}-Q8_0.gguf"]
side = [f + ".sha256" for f in files if f.endswith(".gguf")]
for f in files + side:
    print("UPLOAD", os.path.basename(f), os.path.getsize(f), flush=True)
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo,
                    commit_message=f"bm-889 sft11 final: {os.path.basename(f)}")
info = api.model_info(repo, files_metadata=True)
hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
bad = []
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    b = os.path.basename(f)
    if hub.get(b) != h.hexdigest(): bad.append(b)
    else: print("SHA_OK", b, h.hexdigest()[:16], flush=True)
assert not bad, bad
print("REPO", repo, "private", api.repo_info(repo).private, "commit", info.sha, "files", len(hub))
P
echo "AGQ_OK $(date -u +%FT%TZ)"
