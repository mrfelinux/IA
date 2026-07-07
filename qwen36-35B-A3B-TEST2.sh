#!/bin/bash
set -euo pipefail

llama-server \
   -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_XXS \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 \
  -c $((192*1024)) \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 1024 -ub 512 \
  -t 6 \
  --temp 0.4 --top-p 0.95 --top-k 20 --min-p 0.00\
  --repeat-penalty 1 --presence_penalty 1 \
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 2 \
  --jinja \
  --no-mmap \
  --image-min-tokens 1024 \
  -a qwen3.6-35B-A3B-TEST2 \
  --ctx-checkpoints 16 \
  --cache-reuse 1024 \
  --slot-prompt-similarity 0.10 \
  -n 1024 \
  --reasoning-budget 1024 \
  --reasoning-budget-message "OK, I've thought enough. Let's answer now." \
  --chat-template-kwargs '{"preserve_thinking": true}' 
