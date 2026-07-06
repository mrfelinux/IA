#!/bin/bash
set -euo pipefail

llama-server \
    -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-IQ3_XXS \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 99 \
  -c $((128*1024)) \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -t 6 \
  --mlock \
  --host 0.0.0.0 --port 8080 \
  --metrics \
  -np 1 \
  --jinja \
  --mmap \
  -a qwen3.6-27B-MTP \
  -n 32768 \
