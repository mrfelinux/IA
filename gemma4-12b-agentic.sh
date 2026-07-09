#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
-m /mnt/amazon/modelos/yuxinlu1/gemma4-v2-Q6_K.gguf \
--model-draft /mnt/amazon/modelos/yuxinlu1/MTP/gemma-4-12B-it-MTP-BF16.gguf \
-ngl 99 \
--ctx-size $((128*1024)) \
--spec-type draft-mtp --spec-draft-n-max 4 \
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
--reasoning on \
-a yuxinlu1-gemma4v2
