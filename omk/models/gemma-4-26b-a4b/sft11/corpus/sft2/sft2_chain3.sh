#!/bin/bash
# bm-889 sft2 refill chain (owner 2026-10-30 19:20 CET): no loop turns, every source covered. Generates into $GEN with a
# per-source quota, re-validates, joins the turns of gen_v1r2 that survive the new rules, assembles, renders, stops OUR
# generation server. It NEVER trains: training needs the audit and the owner. Sentinels as chain2, end CHAIN_READY.
# Usage: sft2_chain3.sh JOBS GEN NAME SERVER_PID [generate args...]
set -uo pipefail
G=/mnt/sdc/omk-model-optimizer/g4_recovery; B=$G/sft2_build; PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
JOBS=$1; GEN=$2; NAME=$3; SPID=$4; shift 4
fail(){ echo "CHAIN_FAIL $(date -u +%FT%TZ) $*"; exit 1; }
cd $B || exit 1
echo "CHAIN_START $(date -u +%FT%TZ) jobs $JOBS gen $GEN name $NAME server $SPID args $*"
CUDA_VISIBLE_DEVICES="" $PY sft2_generate.py $JOBS $GEN --url http://172.17.0.1:8261 "$@" > $GEN.log 2>&1 < /dev/null
grep -q "^GEN_OK" $GEN.log || fail "generate: $(tail -2 $GEN.log | tr '\n' ' ' | cut -c1-300)"
echo "CHAIN_GEN_OK $(date -u +%FT%TZ) $(grep ^GEN_OK $GEN.log) $(grep ^GEN_QUOTA_LEFT $GEN.log)"
[ -e ${GEN}r ] && fail "${GEN}r exists"
CUDA_VISIBLE_DEVICES="" $PY sft2_rescore.py $GEN $JOBS ${GEN}r > ${GEN}r.log 2>&1 < /dev/null || fail "rescore"
grep -q "^RESCORE_OK unknown 0" ${GEN}r.log || fail "rescore: $(tail -1 ${GEN}r.log)"
echo "CHAIN_RESCORE_OK $(date -u +%FT%TZ) kept $(wc -l < $GEN/kept.jsonl) -> $(wc -l < ${GEN}r/kept.jsonl)"
mkdir kept_$NAME || fail "kept_$NAME exists"
cat gen_v1r2/kept.jsonl ${GEN}r/kept.jsonl > kept_$NAME/kept.jsonl
CUDA_VISIBLE_DEVICES="" $PY -c "
import json,sys
ids=[json.loads(l)['job'] for l in open('kept_$NAME/kept.jsonl')]
sys.exit(f'duplicate jobs {len(ids)-len(set(ids))}') if len(ids)!=len(set(ids)) else print('JOIN_OK',len(ids))
" || fail "join"
CUDA_VISIBLE_DEVICES="" $PY sft2_assemble.py kept_$NAME/kept.jsonl data_$NAME > data_$NAME.log 2>&1 || fail "assemble"
mkdir render_$NAME || fail "render_$NAME exists"
for c in data_$NAME/*.conv.jsonl; do
  out=render_$NAME/$(basename $c .conv.jsonl).roles.jsonl
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$G/omk_sft2_code $PY -m omk_qad.rollouts --model /mnt/sdc/ml/models/gemma-4-26B-A4B-it \
    --in $c --history-reasoning strip --seq 16384 --pack --out $out >> render_$NAME.log 2>&1 < /dev/null \
    || fail "render $c"
  echo "RENDERED $out $(grep -o '"windows": [0-9]*\|"refused_not_prefix": [0-9]*\|"rows": [0-9]*' ${out%.jsonl}.stats.json | tr '\n' ' ')"
done
echo "CHAIN_RENDER_OK $(date -u +%FT%TZ)"
args=$(ps -o args= -p $SPID 2>/dev/null)
case "$args" in
  *vllm.entrypoints.openai.api_server*"--model /mnt/sdc/ml/models/gemma-4-26B-A4B-it"*"--port 8261"*) ;;
  *) fail "pid $SPID is not our vLLM sft2gen server: $args";;
esac
kill $SPID
for i in $(seq 120); do kill -0 $SPID 2>/dev/null || break; sleep 5; done
kill -0 $SPID 2>/dev/null && fail "server $SPID did not stop"
echo "CHAIN_SERVER_STOPPED $(date -u +%FT%TZ)"
echo "CHAIN_READY $(date -u +%FT%TZ) render_$NAME: audit it; training needs the owner"
