#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1
llama-server \
   -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_XXS \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 \
  --ctx-size $((128*1024)) --rope-scaling yarn  --rope-scale 4  --yarn-orig-ctx 32768 \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 1024 \
  -t 6 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00\
  --presence_penalty 0 \
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 1 \
  --jinja \
  --no-mmap \
  --image-min-tokens 1024 \
  -a qwen3.6-35B-A3B \
  --ctx-checkpoints 16 \
  --cache-reuse 4096 \
  --slot-prompt-similarity 0.10 \
  -n -1 \
  --reasoning-budget 4096 \
  --reasoning-budget-message "OK, I've thought enough. Let's answer now." \
  --chat-template-kwargs '{"preserve_thinking": true}' \

