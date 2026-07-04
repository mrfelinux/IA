llama-server \
-hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
--spec-type draft-mtp --spec-draft-n-max 4 \
-ngl 999 \
--ctx-size $((128*1024)) \
-np 1 \
-fa on \
--mmap \
--mlock \
-t 6 \
--temp 1 --top-p 0.95 --top-k 64 \
--jinja \
--metrics \
--host 0.0.0.0 \

