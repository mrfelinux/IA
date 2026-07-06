#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
-hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_XS \
--spec-type draft-mtp \
--spec-draft-n-max 2 \
-ngl 99 \
--ctx-size $((128*1024)) \
-np 1 \
-fa on \
--mmap \
--mlock \
--threads 6 \
-b 1024 -ub 512 \
-t 6 \
-ctk q4_0 -ctv q4_0 \
--temp 1 --top-p 0.95 --top-k 64 \
--jinja \
--metrics \
--host 0.0.0.0 \
-n -1 
