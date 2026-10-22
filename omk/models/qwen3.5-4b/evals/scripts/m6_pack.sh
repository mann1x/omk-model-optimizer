#!/usr/bin/env bash
# M6 packaging ladder (bm-876, owner 2026-10-22 10:57): the iq2 student's trained IQ2_XXS dense MLP stays as trained;
# only the "packaging" (attention / linear attn / small proj / head / MTP, all trained at IQ2_M-rest types) goes UP in
# precision, using llama.cpp's established ftype rules (layout --rest presets) and the _L convention (head at Q8_0).
# Stock intermediates between IQ2_XXS and Q4_K_M from quantize_gguf.py (same F16, v5 imatrix, KLD base) place every
# file on one size/quality curve. bs2 GPU0 under the solidpc lock wrapper. Existing tools only.
#  0. layouts (MLP fixed IQ2_XXS) + gate: dense_mlp == {IQ2_XXS: 96} in each
#  1. exports of iq2 final.pt + gate: the 96 ffn tensors byte-identical to the trained iq2.gguf
#  2. stock IQ2_M / IQ3_XXS / IQ3_XS / IQ3_M / Q3_K_S / Q3_K_M / Q3_K_L / Q4_K_S
#  3. KLD of the student variants (gguf_kld, same F16 reference)
#  4. HumanEval + HE+ (bm-860 recipe) on every new file
# Sentinels PACK_DONE / PACK_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; RUN=$R/runs/m6_qwen35_4b; CMP=$RUN/compare; KD=$CMP/kld; X=$RUN/pack; H=$RUN/he
M=$R/models/Qwen3.5-4B; TOK=$M; LS=/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python; MO=/srv/ml/envs/envs/omk-modelopt/bin/python; PORT=8161
L=$X/pack_chain.log; mkdir -p $X $H/cache
ts(){ date -u +%H:%M:%SZ; }; say(){ echo "[$(ts)] $*" | tee -a $L; }; fail(){ say "PACK_FAIL $*"; exit 1; }
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_ALLOW_CODE_EVAL=1
VARS="iq3xxs:IQ3_XXS: iq3m:IQ3_M: q3km:Q3_K_M: iq3m_l:IQ3_M:lm_head=q8_0"

say "0. layouts"
for v in $VARS; do
  IFS=: read -r name rest ovr <<< "$v"; mkdir -p $X/$name
  [ -f $X/$name/layout.pt ] || ( cd $R && PYTHONPATH=$R/code/omk-model-optimizer $MO -m omk_qad.layout --model $M \
      --out $X/$name/layout.pt --rest $rest --fixed dense_mlp=iq2_xxs ${ovr:+--override $ovr} > $X/$name/layout.out 2>&1 ) \
    || fail "layout $name"
  $PY - $X/$name/layout.json $name <<'PYL' 2>&1 | tee -a $L || fail "layout gate $name"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["per_class"]["dense_mlp"] == {"IQ2_XXS": 96}, d["per_class"]["dense_mlp"]
print(sys.argv[2], "rest", d["rest"], "overrides", d["overrides"], d["type_counts"], d["sizes_mib"])
PYL
done

say "1. student exports (trained IQ2_XXS core must stay byte-identical)"
for v in $VARS; do
  IFS=: read -r name rest ovr <<< "$v"; G=$X/$name/iq2-pack-$name.gguf
  [ -f $G ] || ( export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
      PYTHONPATH=$R/code/omk-model-optimizer; cd $R
    $MO -W ignore::FutureWarning -m omk_qad.train --phase export --model $M --layout $X/$name/layout.pt --device cuda:0 \
      --load $RUN/iq2/stage1/final.pt --gguf $G --llama-cpp /opt/llama.cpp > $X/$name/export.out 2>&1 ) || fail "export $name"
  grep -h -e '"verified"' -e verify_failures -e type_mismatches -e size_bytes $X/$name/export.out | tr -d '\n' | tee -a $L; echo | tee -a $L
  $MO -W ignore - $RUN/iq2/iq2.gguf $G <<'PYC' 2>&1 | tee -a $L || fail "core gate $name"
import sys
sys.path.insert(0, "/opt/llama.cpp/gguf-py")
from gguf import GGUFReader
a = {t.name: t for t in GGUFReader(sys.argv[1]).tensors}
b = {t.name: t for t in GGUFReader(sys.argv[2]).tensors}
core = [n for n in a if n.startswith("blk.") and ".ffn_" in n and n.endswith(".weight") and a[n].tensor_type.name == "IQ2_XXS"]
assert len(core) == 96, len(core)
bad = [n for n in core if b[n].tensor_type != a[n].tensor_type or bytes(b[n].data) != bytes(a[n].data)]
assert not bad, bad[:5]
moved = sum(1 for n in a if n not in core and (a[n].tensor_type != b[n].tensor_type or bytes(a[n].data) != bytes(b[n].data)))
print("  core gate OK: 96/96 IQ2_XXS ffn tensors byte-identical; packaging tensors changed:", moved)
PYC
done

say "2. stock intermediates (v5 imatrix reused)"
( export LLAMA_CPP_HOME=/mnt/sdc/llama.cpp-311d4211bf OMK_NO_README=1
  $PY /srv/ml/repos/omnimergekit/scripts/quantize_gguf.py --model $M --output-dir $CMP \
    --only IQ2_M,IQ3_XXS,IQ3_XS,IQ3_M,Q3_K_S,Q3_K_M,Q3_K_L,Q4_K_S \
    --no-upload --keep-local --base-model-id Qwen/Qwen3.5-4B > $CMP/quantize_pack.log 2>&1 ) || fail "quantize_gguf (quantize_pack.log)"
grep -E "imatrix already|reusing|sanity|KLD" $CMP/quantize_pack.log | tail -20 | tee -a $L

say "3. KLD of the student variants"
for v in $VARS; do
  IFS=: read -r name rest ovr <<< "$v"
  $PY - "$KD" "$X/$name/iq2-pack-$name.gguf" "M6-iq2-pack-$name" <<'PYK' 2>&1 | tee -a $L || fail "KLD $name"
import json, sys
from pathlib import Path
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
import gguf_kld
kd, g, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
meta_p = next(kd.glob("kld_base_*.meta.json"))
basis = json.loads(meta_p.read_text())
basis["base_file"] = str(meta_p.with_name(meta_p.name.replace(".meta.json", ".bin")))
ppl = gguf_kld.find_perplexity({"quantize": "/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-quantize"}, build_if_missing=False)
row = gguf_kld.measure(ppl, g, tag, basis, kd / "wikitext-2-raw/wiki.test.raw")
gguf_kld.append_row(kd, row)
print(tag, g.stat().st_size, row["kld_mean"][0], row["same_top_p_pct"][0], row["ppl_q"][0])
PYK
done

say "4. HumanEval + HE+ (bm-860 recipe)"
he(){ # name, gguf
  local name=$1 g=$2 srv n=0 t rc o
  $LS -m $g --alias $name --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c 8192 -np 2 > $H/srv.$name.pack.log 2>&1 &
  srv=$!
  until curl -sf http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok \
        && curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -qe "$name" -e "$(basename $g)"; do
    kill -0 $srv 2>/dev/null || { say "FAIL server $name died"; return 1; }
    n=$((n+1)); [ $n -gt 120 ] && { say "FAIL server $name not ready"; kill $srv; return 1; }; sleep 5
  done
  for t in humaneval humaneval_plus; do
    o=$H/out_$name; [ $t = humaneval_plus ] && o=$H/out_${name}_hep
    $(dirname $PY)/lm_eval --model local-completions \
      --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TOK,max_gen_toks=2048" \
      --tasks $t --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
      --use_cache $H/cache/$name.$t --log_samples --output_path $o > $H/lm_eval.$name.$t.log 2>&1
    rc=$?
    say "$name $t rc=$rc $($PY /srv/ml/scripts/bonsai4b_he_samples.py $o 2>&1 | tail -1)"
  done
  kill $srv; wait $srv 2>/dev/null
}
for v in $VARS; do IFS=: read -r name rest ovr <<< "$v"; he m6-iq2-pack-$name $X/$name/iq2-pack-$name.gguf; done
for q in IQ2_M IQ3_XXS IQ3_XS IQ3_M Q3_K_S Q3_K_M Q3_K_L Q4_K_S; do
  he qwen3.5-4b-$(echo $q | tr 'A-Z' 'a-z' | tr -d _) $CMP/Qwen3.5-4B-$q.gguf
done
say "PACK_DONE"
