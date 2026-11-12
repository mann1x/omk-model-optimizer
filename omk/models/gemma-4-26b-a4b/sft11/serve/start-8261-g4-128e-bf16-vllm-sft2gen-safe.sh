#!/bin/bash
# start-8261-g4-128e-bf16-vllm-sft2gen-safe.sh -- bm-889 sft2 generation (2026-10-30): the UNTOUCHED base gemma-4-26B-A4B-it
# (HF bf16 safetensors, the same weights as the BF16 GGUF the gate serves) on vLLM 0.20.2 for batch throughput.
# Template = the gate server's (6a1015c4 = Google ae53464b without its trailing newline), passed explicitly.
# --enforce-eager: no CUDA graphs. vLLM 0.20.2 gemma4.py still captures per_expert_scale in a closure, the pattern of
# the 2026-05 bad-routing bug under graph capture (memory feedback_vllm_gemma4_silentempty_rca); training data must not
# carry wrong-expert tokens, so the graph path is not used.
# gemma4_safe (2026-10-30 10:50Z): stock _parse_gemma4_array loops forever on a stray "]" element and blocked the API
# server event loop at 09:59Z (py-spy); the plugin replaces it with a progress-guarded copy (fuzz: 19,948 identical, 59
# stock hangs now terminate; test_gemma4_safe.py).
# Context 16384 = the sft2 training window. Bound to docker0 172.17.0.1 only. Claim bs2/gpu0 before start.
# Start: setsid nohup bash /srv/ml/start-8261-g4-128e-bf16-vllm-sft2gen-safe.sh >> /srv/ml/opencoti-dev/logs/8261-g4-128e-bf16-vllm-sft2gen-safe.log 2>&1 < /dev/null &
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
exec /srv/ml/envs/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/sdc/ml/models/gemma-4-26B-A4B-it --served-model-name gemma-4-26b-a4b-it-bf16 \
  --host 172.17.0.1 --port 8261 --dtype bfloat16 --enforce-eager \
  --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --chat-template /mnt/sdc/omk-model-optimizer/g4_recovery/sft2_build/served_template_6a1015c4.jinja \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4_safe --tool-parser-plugin /mnt/sdc/omk-model-optimizer/g4_recovery/sft2_build/gemma4_safe_tool_parser.py
