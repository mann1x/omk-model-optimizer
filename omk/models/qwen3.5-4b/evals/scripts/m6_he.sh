#!/usr/bin/env bash
# M6 (bm-876, owner 2026-10-22 08:46 "run the humaneval now"), bs2 GPU0 under the solidpc lock wrapper.
#  1. PQ2_0 KLD with opencoti's tool (mail #1779). Validity control first: the same tool on stock F16 (expect ~0)
#     and stock Q4_K_M (expect ~0.0271 as stock llama.cpp measured) against our 311d4211bf F16 base; tern only if both pass.
#  2. HumanEval-164 with the bm-860 recipe (bonsai4b_he_chain3.sh): lm-eval humaneval, local-completions, num_concurrent=2,
#     max_gen_toks=2048, greedy, tokenizer = Qwen3.5-4B dir. Stock llama-server: F16, Q4_K_M, IQ2_S, iq2 student.
#     opencoti dev engine: tern student + Q4_K_M as the engine control.
# Sentinels M6HE_DONE / M6HE_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; RUN=$R/runs/m6_qwen35_4b; CMP=$RUN/compare; KD=$CMP/kld; H=$RUN/he; L=$H/he_chain.log
TOK=$R/models/Qwen3.5-4B; LS=/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-server
OCT=/mnt/sdc/opencoti-tools/kld-2610220831001; OCE=/srv/ml/opencoti-dev/opencoti-0.10.5-c13-2610220831001
PY=/root/anaconda3/envs/omnimergekit/bin/python; PORT=8161
BASE=$KD/kld_base_Qwen3.5-4B-F16_c512_n100.bin; WIKI=$KD/wikitext-2-raw/wiki.test.raw
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_ALLOW_CODE_EVAL=1
mkdir -p $H/cache; ts(){ date -u +%H:%M:%SZ; }; say(){ echo "[$(ts)] $*" | tee -a $L; }

say "1. opencoti PQ2_0 KLD (validity control first)"
ockld(){ # tag, gguf -> prints "tag kld same_top ppl gpu=<n CUDA0 lines>"
  env HOME=$OCT/home $OCT/perplexity -v -m $2 -f $WIKI -c 512 --chunks 100 -ngl 99 -fa on \
      --kl-divergence-base $BASE --kl-divergence > $H/ockld.$1.log 2>&1
  $PY - $H/ockld.$1.log $1 <<'PYK'
import sys
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
import gguf_kld
t = open(sys.argv[1], errors="replace").read()
try:
    r = gguf_kld.parse_kld_log(t)
    print(sys.argv[2], r["kld_mean"][0], r["same_top_p_pct"][0], r["ppl_q"][0], "gpu_lines=%d" % t.count("CUDA0"))
except RuntimeError as e:
    print(sys.argv[2], "PARSE_FAIL", e)
PYK
}
F16C=$(ockld f16 $CMP/Qwen3.5-4B-F16.gguf); say "control $F16C"
Q4C=$(ockld q4km $CMP/Qwen3.5-4B-Q4_K_M.gguf); say "control $Q4C (stock tool: 0.027055 / 92.404 / 9.8703)"
ok=$($PY -c "import sys; f=sys.argv[1].split(); q=sys.argv[2].split()
print(int(len(f)>3 and len(q)>3 and 'PARSE_FAIL' not in f+q and float(f[1])<0.002 and abs(float(q[1])-0.027055)<0.002))" "$F16C" "$Q4C")
if [ "$ok" = 1 ]; then say "control PASS -> tern real file: $(ockld tern $RUN/tern/tern.gguf)"
else say "control FAIL -> tern real-file KLD skipped (logs $H/ockld.*.log)"; fi

say "2. HumanEval-164 (bm-860 recipe)"
he(){ # name, gguf, engine(stock|oc)
  local name=$1 g=$2 eng=$3 srv n=0
  if [ $eng = stock ]; then
    $LS -m $g --alias $name --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c 8192 -np 2 > $H/srv.$name.log 2>&1 &
  else
    env HOME=$OCT/home $OCE --server --nologo --host 127.0.0.1 --port $PORT -m $g --alias $name --gpu nvidia \
      -ngl 99 -fa on -c 8192 -np 2 --no-webui > $H/srv.$name.log 2>&1 &
  fi
  srv=$!
  until curl -sf http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok \
        && curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -qe "$name" -e "$(basename $g)"; do
    kill -0 $srv 2>/dev/null || { say "FAIL server $name died (srv.$name.log)"; return 1; }
    n=$((n+1)); [ $n -gt 120 ] && { say "FAIL server $name not ready"; kill $srv; return 1; }; sleep 5
  done
  say "serving $name ($eng, $(basename $g))"
  $(dirname $PY)/lm_eval --model local-completions \
    --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TOK,max_gen_toks=2048" \
    --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$name --log_samples --output_path $H/out_$name > $H/lm_eval.$name.log 2>&1
  local rc=$?
  kill $srv; wait $srv 2>/dev/null
  say "$name rc=$rc $($PY /srv/ml/scripts/bonsai4b_he_samples.py $H/out_$name 2>&1 | tail -1)"
}
he qwen3.5-4b-f16 $CMP/Qwen3.5-4B-F16.gguf stock
he qwen3.5-4b-q4km $CMP/Qwen3.5-4B-Q4_K_M.gguf stock
he qwen3.5-4b-iq2s $CMP/Qwen3.5-4B-IQ2_S.gguf stock
he m6-iq2-student $RUN/iq2/iq2.gguf stock
he m6-tern-student $RUN/tern/tern.gguf oc
he qwen3.5-4b-q4km-oc $CMP/Qwen3.5-4B-Q4_K_M.gguf oc
say "M6HE_DONE"
