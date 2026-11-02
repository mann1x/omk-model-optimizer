#!/bin/bash
cd /mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build
PY=/mnt/sdc/omk-model-optimizer/venv/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cat jobs.txt | xargs -P 8 -L 1 bash -c 'n=$(basename $1 .roles.jsonl); '"$PY"' render_sft4.py "$1" "$2" "$3" "$4" > logs/$n.log 2>&1 && echo "OK $n" || echo "FAIL $n $(tail -2 logs/$n.log | tr "\n" " ")"' _ 
echo RENDER_ALL_DONE
