#!/usr/bin/env bash
# bm-885: build mix_v2 (stage-2 corpus, 20.0M train tokens, no reuse of mix_v1 / the stage-1 cache).
set -euo pipefail
R=/mnt/sdc/omk-model-optimizer; W=$R/runs/m6s2; PY=/root/anaconda3/envs/omk-modelopt/bin/python
cd $W
$PY build_mix.py --name mix_v2 --seed 20261022 --exclude $R/data/mix_v1/manifest.json \
  --only opencode,agentic,swe_agentless,cp_reasoning,exercism_sql \
  --budget opencode=4.5e6,agentic=3.0e6,swe_agentless=1.5e6,cp_reasoning=1.5e6,exercism_sql=1.0e6
PYTHONPATH=$R/code/omk-model-optimizer $PY fresh_tail.py --cache $R/runs/m6_qwen35_4b/cache \
  --tokenizer $R/models/Qwen3.5-4B --out $R/data/mix_v2/fresh \
  --take corpus_v1/train_code.jsonl=2.0e6,corpus_v1/train_tech.jsonl=1.0e6,corpus_v1/train_general.jsonl=1.0e6,corpus_v1/agent_train.jsonl=0.5e6,fineweb/sample_10BT_000_00000.parquet=4.0e6
echo MIX_V2_DONE
