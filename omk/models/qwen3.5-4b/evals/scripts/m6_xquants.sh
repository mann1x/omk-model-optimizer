#!/usr/bin/env bash
# M6 extra quants (bm-876, owner 2026-10-22 09:12 "only IQ grids, matching stock IQ2_XXS, and IQ1_M / IQ2_XS"),
# bs2 GPU0 under the solidpc lock wrapper. Existing tools only:
#  0. layout control: omk_qad.layout CLI re-plans the trained iq2 layout (rest IQ2_M, dense_mlp IQ2_XXS) -> must equal
#     iq2/layout.json type_counts + per_class; then the same plan with dense_mlp IQ1_M / IQ2_XS.
#  1. omk_qad.train --phase export of the iq2 student's final.pt at those layouts (its own trained imatrix).
#  2. stock quantize_gguf.py IQ2_XXS / IQ1_M / IQ2_XS into compare/ (reuses F16, v5 imatrix.dat, KLD base).
#  3. KLD of the two student exports on the same F16 reference (gguf_kld).
#  4. HumanEval (bm-860 recipe) on the 5 new files; HE+ (humaneval_plus, same recipe) on the 5 new + the 6 existing.
# Sentinels XQ_DONE / XQ_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; RUN=$R/runs/m6_qwen35_4b; CMP=$RUN/compare; KD=$CMP/kld; X=$RUN/xq; H=$RUN/he
M=$R/models/Qwen3.5-4B; TOK=$M; LS=/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-server
OCT=/mnt/sdc/opencoti-tools/kld-2610220831001; OCE=/srv/ml/opencoti-dev/opencoti-0.10.5-c13-2610220831001
PY=/root/anaconda3/envs/omnimergekit/bin/python; MO=/srv/ml/envs/envs/omk-modelopt/bin/python; PORT=8161
L=$X/xq_chain.log; mkdir -p $X $H/cache
ts(){ date -u +%H:%M:%SZ; }; say(){ echo "[$(ts)] $*" | tee -a $L; }; fail(){ say "XQ_FAIL $*"; exit 1; }
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_ALLOW_CODE_EVAL=1

say "0. layouts"
lay(){ # dir, dense_mlp type
  mkdir -p $X/$1
  [ -f $X/$1/layout.pt ] || ( cd $R && PYTHONPATH=$R/code/omk-model-optimizer $MO -m omk_qad.layout --model $M \
      --out $X/$1/layout.pt --rest IQ2_M --fixed dense_mlp=$2 > $X/$1/layout.out 2>&1 ) || fail "layout $1"
}
lay ctrl iq2_xxs; lay iq1m iq1_m; lay iq2xs iq2_xs
$PY - $RUN/iq2/layout.json $X/ctrl/layout.json $X/iq1m/layout.json $X/iq2xs/layout.json <<'PYL' || fail "layout control"
import json, sys
a, b, *new = (json.load(open(p)) for p in sys.argv[1:])
for k in ("rest", "fixed", "overrides", "type_counts", "per_class"):
    if a[k] != b[k]:
        sys.exit(f"control differs on {k}: trained {a[k]} vs replanned {b[k]}")
print("control OK: replanned == trained", a["type_counts"])
for n in new:
    print(n["fixed"], n["type_counts"], n["sizes_mib"])
PYL
cat $X/*/layout.out | grep -iE "warn|error" | head -5

say "1. student exports"
for v in iq1m iq2xs; do
  G=$X/$v/iq2-$v.gguf
  [ -f $G ] || ( export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
      PYTHONPATH=$R/code/omk-model-optimizer; cd $R
    $MO -W ignore::FutureWarning -m omk_qad.train --phase export --model $M --layout $X/$v/layout.pt --device cuda:0 \
      --load $RUN/iq2/stage1/final.pt --gguf $G --llama-cpp /opt/llama.cpp > $X/$v/export.out 2>&1 ) || fail "export $v"
  grep -h -e '"verified"' -e verify_failures -e type_mismatches -e size_bytes -e with_imatrix $X/$v/export.out | tr -d '\n'; echo
done

say "2. stock IQ2_XXS / IQ1_M / IQ2_XS (v5 imatrix reused)"
( export LLAMA_CPP_HOME=/mnt/sdc/llama.cpp-311d4211bf OMK_NO_README=1
  $PY /srv/ml/repos/omnimergekit/scripts/quantize_gguf.py --model $M --output-dir $CMP --only IQ2_XXS,IQ1_M,IQ2_XS \
    --no-upload --keep-local --base-model-id Qwen/Qwen3.5-4B > $CMP/quantize_xq.log 2>&1 ) || fail "quantize_gguf (quantize_xq.log)"
grep -E "imatrix already|reusing|sanity|KLD" $CMP/quantize_xq.log | tail -10 | tee -a $L

say "3. KLD of the student exports"
for v in iq1m iq2xs; do
  $PY - "$KD" "$X/$v/iq2-$v.gguf" "M6-iq2@$v" <<'PYK' 2>&1 | tee -a $L || fail "KLD $v"
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
print(tag, g.stat().st_size, row["kld_mean"], row["same_top_p_pct"], row["ppl_q"])
PYK
done

say "4. HumanEval + HE+ (bm-860 recipe)"
he(){ # name, gguf, engine(stock|oc), tasks (space list)
  local name=$1 g=$2 eng=$3 srv n=0 t rc
  if [ $eng = stock ]; then
    $LS -m $g --alias $name --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c 8192 -np 2 > $H/srv.$name.xq.log 2>&1 &
  else
    env HOME=$OCT/home $OCE --server --nologo --host 127.0.0.1 --port $PORT -m $g --alias $name --gpu nvidia \
      -ngl 99 -fa on -c 8192 -np 2 --no-webui > $H/srv.$name.xq.log 2>&1 &
  fi
  srv=$!
  until curl -sf http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok \
        && curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -qe "$name" -e "$(basename $g)"; do
    kill -0 $srv 2>/dev/null || { say "FAIL server $name died"; return 1; }
    n=$((n+1)); [ $n -gt 120 ] && { say "FAIL server $name not ready"; kill $srv; return 1; }; sleep 5
  done
  for t in $4; do
    local o=$H/out_$name; [ $t = humaneval_plus ] && o=$H/out_${name}_hep
    $(dirname $PY)/lm_eval --model local-completions \
      --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TOK,max_gen_toks=2048" \
      --tasks $t --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
      --use_cache $H/cache/$name.$t --log_samples --output_path $o > $H/lm_eval.$name.$t.log 2>&1
    rc=$?
    say "$name $t rc=$rc $($PY /srv/ml/scripts/bonsai4b_he_samples.py $o 2>&1 | tail -1)"
  done
  kill $srv; wait $srv 2>/dev/null
}
B="humaneval humaneval_plus"
he qwen3.5-4b-iq2xxs $CMP/Qwen3.5-4B-IQ2_XXS.gguf stock "$B"
he m6-iq2-student-iq1m $X/iq1m/iq2-iq1m.gguf stock "$B"
he qwen3.5-4b-iq1m $CMP/Qwen3.5-4B-IQ1_M.gguf stock "$B"
he m6-iq2-student-iq2xs $X/iq2xs/iq2-iq2xs.gguf stock "$B"
he qwen3.5-4b-iq2xs $CMP/Qwen3.5-4B-IQ2_XS.gguf stock "$B"
he qwen3.5-4b-f16 $CMP/Qwen3.5-4B-F16.gguf stock humaneval_plus
he qwen3.5-4b-q4km $CMP/Qwen3.5-4B-Q4_K_M.gguf stock humaneval_plus
he qwen3.5-4b-iq2s $CMP/Qwen3.5-4B-IQ2_S.gguf stock humaneval_plus
he m6-iq2-student $RUN/iq2/iq2.gguf stock humaneval_plus
he m6-tern-student $RUN/tern/tern.gguf oc humaneval_plus
he qwen3.5-4b-q4km-oc $CMP/Qwen3.5-4B-Q4_K_M.gguf oc humaneval_plus
say "XQ_DONE"
