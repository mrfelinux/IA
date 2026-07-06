#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
-hf google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0 \
-ngl 99 \
--ctx-size $((128*1024)) \
-np 1 \
-fa on \
--mmap \
--mlock \
--threads 6 \
-b 2048 -ub 1024 \
-t 6 \
-ctk q4_0 -ctv q4_0 \
--temp 1 --top-p 0.95 --top-k 64 \
--repeat-penalty 1 \
--jinja \
--metrics \
--host 0.0.0.0 \
-n -1 \
--reasoning-preserve
