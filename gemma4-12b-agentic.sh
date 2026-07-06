#!/bin/bash
set -euo pipefail

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

llama-server \
-hf yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q6_K \
-md yuxinlu1/MTP/gemma-4-12B-it-MTP-Q8_0.gguf \
-ngl 99 \
--ctx-size $((128*1024)) \
--spec-type draft-mtp --spec-draft-n-max 4 \
-np 1 \
-fa on \
--no-mmap \
--mlock \
-t 6 \
--temp 1 --top-p 0.95 --top-k 64 \
--jinja \
--metrics \
--host 0.0.0.0 \
--reasoning on
