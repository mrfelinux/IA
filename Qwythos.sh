#!/bin/bash
set -euo pipefail

llama-server \
  -hf empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --jinja \
  --metrics \
  --host 0.0.0.0 \
  --temp 0.6 --top-p 0.95 --top-k 20  \
  --repeat-penalty 1  \
  -t 6  \
  -ngl 99 \
  -c $((128*1024)) \
  --flash-attn on \
  --parallel 1 \
  --mmap \
  --n-predict 16384 \
  --reasoning on 

