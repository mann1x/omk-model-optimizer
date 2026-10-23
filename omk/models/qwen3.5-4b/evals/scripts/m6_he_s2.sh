#!/usr/bin/env bash
# M6 stage 2 (bm-885, owner 2026-10-23 11:07 "run HE on it"): HumanEval-164 + HumanEval+ on the stage-2 B GGUF
# (iq2_s2_mix_v2_lr3e6.gguf), the he() recipe of m6_he.sh verbatim (stock llama-server 311d4211bf, local-completions,
# num_concurrent 2, max_gen_toks 2048, greedy, tokenizer Qwen3.5-4B, sqlite cache, log_samples). Stage-1 reference:
# m6-iq2-student HE 63/164 38.41 %, HE+ 0.3171. Sentinels M6HES2_DONE / M6HES2_FAIL.
set -uo pipefail
R=/mnt/sdc/omk-model-optimizer; RUN=$R/runs/m6_qwen35_4b; H=$RUN/he; L=$H/he_chain.log
TOK=$R/models/Qwen3.5-4B; LS=/mnt/sdc/llama.cpp-311d4211bf/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python; PORT=8161
G=$RUN/iq2/stage2/mix_v2_lr3e6/iq2_s2_mix_v2_lr3e6.gguf; name=m6-iq2-s2-mix_v2_lr3e6
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_ALLOW_CODE_EVAL=1
ts(){ date -u +%H:%M:%SZ; }; say(){ echo "[$(ts)] $*" | tee -a $L; }
[ -z "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)" ] || { say "M6HES2_FAIL GPU0 busy"; exit 1; }
$LS -m $G --alias $name --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c 8192 -np 2 > $H/srv.$name.log 2>&1 &
srv=$!; n=0
until curl -sf http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok \
      && curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q -e "$name"; do
  kill -0 $srv 2>/dev/null || { say "M6HES2_FAIL server died (srv.$name.log)"; exit 1; }
  n=$((n+1)); [ $n -gt 120 ] && { say "M6HES2_FAIL server not ready"; kill $srv; exit 1; }; sleep 5
done
say "serving $name (stock, $(basename $G)) -- stage-2 HE + HE+"
for task in humaneval humaneval_plus; do
  sfx=$([ $task = humaneval ] && echo "" || echo _hep); csfx=$([ $task = humaneval ] && echo "" || echo .humaneval_plus)
  lsfx=$([ $task = humaneval ] && echo "" || echo .humaneval_plus)
  $(dirname $PY)/lm_eval --model local-completions \
    --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TOK,max_gen_toks=2048" \
    --tasks $task --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$name$csfx --log_samples --output_path $H/out_$name$sfx > $H/lm_eval.$name$lsfx.log 2>&1
  rc=$?
  say "$name $task rc=$rc $(grep -E "^\|$task " $H/lm_eval.$name$lsfx.log | head -1) | $($PY /srv/ml/scripts/bonsai4b_he_samples.py $H/out_$name$sfx 2>&1 | tail -1)"
done
kill $srv; wait $srv 2>/dev/null
say "M6HES2_DONE"
