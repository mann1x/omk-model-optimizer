#!/bin/bash
cd /mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export MLAST_TPL=/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/chat_template_mlast2.jinja CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cat jobs_v3_mlast2.txt | xargs -P 8 -L 1 bash -c 'n=$(basename $1 .roles.jsonl); '"$PY"' render_sft4_mlast.py "$1" "$2" "$3" "$4" > logs/v3m2_$n.log 2>&1 && echo "OK $n" || echo "FAIL $n $(tail -2 logs/v2_$n.log | tr "\n" " ")"' _ 
echo RENDER_ALL_DONE
