#!/bin/bash
set -euo pipefail

llama-server \
  -hf unsloth/Qwen-AgentWorld-35B-A3B-GGUF:UD-IQ2_M \
  -ngl 999 \
  -c 120000 \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 2048 \
  -t 6 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00\
  --repeat-penalty 1.1 \
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 1 \
  --jinja \
  --prio 3 \
  --prio-batch 3 \
  --poll 100 \
  --poll-batch 1 \
  --no-mmap \
  --image-min-tokens 1024 \
  -a qwen3.6-35B-A3B \
  --reasoning off \
  --ctx-checkpoints 16 \
  --cache-reuse 1024 \
  --slot-prompt-similarity 0.10 \
  -n -1 \
  --reasoning-budget 1024 \
  --reasoning-budget-message "OK, I've thought enough. Let's answer now." \
  --chat-template-kwargs '{"preserve_thinking": true}'
