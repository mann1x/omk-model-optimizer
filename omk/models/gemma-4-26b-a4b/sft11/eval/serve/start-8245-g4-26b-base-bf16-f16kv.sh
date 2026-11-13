#!/bin/bash
# start-8245-g4-26b-base-bf16-f16kv.sh -- owner 2026-11-12 22:41/22:43 CET: SDK .277 head-to-head, base on GPU0 :8245 vs bm-889 sft11-final on GPU1 :8244,
# same sampler (temp 0.8, top_p 0.95, top_k 64, min_p 0, presence 0.025), "use for serving f16 kv cache".
# = start-8244-bm889-sft11-sfinal-bf16.sh (engine c14 r15, template, drafter, parallel, sampler, budget message verbatim) with
# model = the untouched base gemma-4-26B-A4B-it BF16 eos106 GGUF, port 8245, GPU 0, sole alias = the automaticum model tag, KV cache f16 everywhere (was kvarn3 + SWA q8_0; the engine allows SWA overrides only with KVarN, f16 covers both). Claim bs2/gpu0 first.
# Start: setsid nohup bash /srv/ml/start-8245-g4-26b-base-bf16-f16kv.sh >> /srv/ml/opencoti-dev/logs/start-8245-g4-26b-base-bf16-f16kv.log 2>&1 < /dev/null &
set -euo pipefail
ulimit -c 31457280  # 30 GiB (KiB units)
export CUDA_VISIBLE_DEVICES=0
export OPENCOTI_MTP_TOKEMBD_GPU=1
M=/mnt/sdc/opencoti-models/gemma-4-26B-A4B/it-bf16/gemma-4-26B-A4B-it-BF16-eos106.gguf
D=/srv/ml/models/drafters-hf/gemma-4-26B-A4B-it-assistant.Q8_0.gguf
BUDGET_MSG=$'\n\nI have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.\nI\'ll be more terse and concise now and maybe I need to consider a different approach.\n'
exec /srv/ml/opencoti-rel/c14-r15/opencoti-llamafile-0.10.5-c14-bare.llamafile \
  --server --nologo --host 127.0.0.1 --port 8245 --slots \
  -m "$M" \
  -a gemma-4-26b-a4b-it-bf16-eos106 \
  --spec-type draft-assistant \
  --mtp-head "$D" \
  -ngld 99 --spec-draft-n-max 2 \
  --chat-template-file /srv/ml/templates/gemma4-google-canonical-20260709.jinja \
  --gpu nvidia -ngl 99 --flash-attn on -b 1024 -ub 1024 \
  -c 1048576 -kvu \
  -ctk f16 -ctv f16 \
  --parallel 4 --max-parallel 32 --max-parallel-tps-floor 5 \
  --polykv-max-pools 8 --no-cache-idle-slots \
  --presence-penalty 0.025 \
  --temp 0.8 --top-p 0.95 --top-k 64 --min-p 0 \
  --reasoning-budget-message "$BUDGET_MSG"
