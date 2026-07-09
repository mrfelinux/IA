#!/bin/bash
set -euo pipefail

llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_XXS \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 999 \
  -fa on \
  -fit off \
  -c $((128*1024)) \
  --reasoning on\
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --cache-type-k-draft q8_0\
  --cache-type-v-draft q8_0\
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  -np 1 \
  --cache-idle-slots \
  --kv-unified \
  -a qwen3.6-35B-A3B-nicho \
  --no-mmap \
  --host 0.0.0.0 \
  --metrics
