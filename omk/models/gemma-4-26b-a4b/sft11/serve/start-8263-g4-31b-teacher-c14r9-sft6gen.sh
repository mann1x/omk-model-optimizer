#!/bin/bash
# start-8263-g4-31b-teacher-c14r9-sft6gen.sh -- bm-889 sft6 C8 (owner 2026-11-04 12:12 CET "start the C8 kimi
# regeneration with the 31B"): Gemma-4 31B-it BF16 (the run4 teacher GGUF, 61,413,189,856 B, sha in its folder)
# served by opencoti c14 r9 (build 2611031149001) on GPU1 for sft2_generate.py: 4 slots x 12288 ctx (the owner
# set 12k as the hard max per row), thinking on, reasoning split out (deepseek), no budget. Port 8263, not the
# gate port. Claim bs2/gpu1 before start.
set -euo pipefail
ulimit -c 31457280
export CUDA_VISIBLE_DEVICES=1
M=/mnt/sdc/ml/gguf/g4-31b-teacher/gemma-4-31B-it-BF16.gguf
exec /srv/ml/opencoti-rel/c14-r9/opencoti-llamafile-0.10.5-c14-bare.llamafile \
  --server --nologo --host 172.17.0.1 --port 8263 --slots \
  -m "$M" -a gemma-4-31b-it-bf16 \
  --jinja --chat-template-kwargs '{"enable_thinking": true}' --reasoning-format deepseek --reasoning-budget -1 \
  --gpu nvidia -ngl 99 --flash-attn on -b 1024 -ub 1024 \
  -c 49152 -np 4
