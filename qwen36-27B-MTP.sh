#!/bin/bash
set -euo pipefail
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-IQ3_XXS \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 \
  --ctx-size $((128*1024)) --rope-scaling yarn  --rope-scale 4  --yarn-orig-ctx 32768 \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 1024 \
  -t 6 \
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 1 \
  --jinja --chat-template-file templates/chat_template-froggeric-qwen.jinja\
  --mmap \
  -a qwen3.6-27B-MTP \
  -n -1 \
  --reasoning-budget 512 \
  --reasoning-budget-message "OK, I've thought enough. Let's answer now." \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --image-min-tokens 1024

