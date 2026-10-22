#!/usr/bin/env bash
# M6 comparison (bm-876/878, owner 08:25): existing tools only, on bs2 GPU0 (lock held by the solidpc wrapper).
#  1. omk_qad.multi export + report: each student's GGUF at its trained format (iq2 IQ2_XXS-class, tern PQ2_0)
#  2. stock quantize_gguf.py on Qwen3.5-4B: Q4_K_M + IQ2_S, default v5-calib imatrix (128 chunks), KLD vs F16
#  3. KLD of the iq2 student GGUF on the same F16 reference (gguf_kld)
# Sentinels M6CMP_DONE / M6CMP_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; SPEC=$R/runs/qwen35_4b_smoke.yaml; RUN=$R/runs/m6_qwen35_4b; CMP=$RUN/compare
LC=/mnt/sdc/llama.cpp-311d4211bf; OMKS=/srv/ml/repos/omnimergekit/scripts; PY=/root/anaconda3/envs/omnimergekit/bin/python
fail(){ echo "M6CMP_FAIL $*"; exit 1; }
say(){ echo ">>> $(date -u +%FT%TZ) $*"; }
mkdir -p $CMP

say "1. omk_qad.multi export + report"
( export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
  export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$R/code/omk-model-optimizer
  cd $R
  for ph in export report; do
    /srv/ml/envs/envs/omk-modelopt/bin/python -W ignore::FutureWarning -m omk_qad.multi --spec $SPEC $ph || exit 1
  done ) || fail "omk_qad.multi export/report (logs in $RUN/logs/*.export.out)"
grep -h -e verified -e verify_failures -e type_mismatches -e size_bytes $RUN/logs/iq2.export.out $RUN/logs/tern.export.out

say "2. quantize_gguf Q4_K_M + IQ2_S (v5 calib imatrix)"
export LLAMA_CPP_HOME=$LC CUDA_VISIBLE_DEVICES=0 OMK_NO_README=1
$PY $OMKS/quantize_gguf.py --model $R/models/Qwen3.5-4B --output-dir $CMP --only Q4_K_M,IQ2_S \
    --no-upload --keep-local --base-model-id Qwen/Qwen3.5-4B > $CMP/quantize.log 2>&1 || fail "quantize_gguf (see $CMP/quantize.log)"
grep -E "sanity|KLD|imatrix" $CMP/quantize.log | tail -12

say "3. KLD of the iq2 student GGUF on the same F16 reference"
IQ2=$RUN/iq2/iq2.gguf; [ -f "$IQ2" ] || fail "no iq2 GGUF under $RUN/iq2"
$PY - "$CMP/kld" "$IQ2" <<'PYK' || fail "iq2 KLD"
import json, sys
from pathlib import Path
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
import gguf_kld
kd, g = Path(sys.argv[1]), Path(sys.argv[2])
meta_p = next(kd.glob("kld_base_*.meta.json"))
basis = json.loads(meta_p.read_text())
basis["base_file"] = str(meta_p.with_name(meta_p.name.replace(".meta.json", ".bin")))  # not with_suffix: .meta.bin
ppl = gguf_kld.find_perplexity({"quantize": "/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-quantize"}, build_if_missing=False)
row = gguf_kld.measure(ppl, g, "M6-iq2", basis, kd / "wikitext-2-raw/wiki.test.raw")
gguf_kld.append_row(kd, row)
print("M6-iq2", g.stat().st_size, row["kld_mean"], row["same_top_p_pct"], row["ppl_q"])
PYK
echo "M6CMP_DONE $(date -u +%FT%TZ)"
