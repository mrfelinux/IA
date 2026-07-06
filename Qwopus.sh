#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
  -hf Jackrong/Qwopus3.5-9B-Coder-MTP-GGUF:Q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 \
  -c $((128*1024)) \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 1024 -ub 512 \
  -t 6 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00\
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 1 \
  --jinja \
  --no-mmap \
  --image-min-tokens 1024 \
  -a Jackrong/Qwopus3.5-9B-Coder-MTP \
  --cache-reuse 4096 \
  --slot-prompt-similarity 0.10 \
  -n -1 \
  --reasoning-budget 1024 \
  --reasoning-budget-message "OK, I've thought enough. Let's answer now." \
  --chat-template-kwargs '{"preserve_thinking": true}' 
