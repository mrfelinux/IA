#!/bin/bash
set -euo pipefail
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1
llama-server \
-hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
--spec-type draft-mtp --spec-draft-n-max 4 \
-ctk q4_0 -ctv q4_0 \
-b 1024 -ub 512 \
-ngl 999 \
--ctx-size $((128*1024)) \
-np 1 \
-fa on \
-t 6 \
--temp 1 --top-p 0.95 --top-k 64 \
--repeat-penalty 1 --presence_penalty 1 \
--ctx-checkpoints 16 \
--cache-reuse 1024 \
--reasoning-budget 1024 \
--jinja \
--metrics \
--host 0.0.0.0 \
-a unsloth-gemma4-it-qat2

