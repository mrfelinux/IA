#!/bin/bash
set -euo pipefail

# Script para iniciar llama-server con Ornith-1.0-9B-GGUF (Q4_K_M)
# Optimizado para 16GB VRAM y uso con Hermes Agent (256K ctx, batch optimizado)
# Incluye optimizaciones para AMD RX 9060 XT

# --- NUEVO: Habilitar optimizaciones para AMD ---
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

# Configuración del modelo y servidor
MODEL_HF="protoLabsAI/Ornith-1.0-9B-MTP-GGUF:Q4_K_M"

# Alternativa de mayor calidad: "deepreinforce-ai/Ornith-1.0-9B-GGUF:UD-Q4_K_XL"
PORT=8080
HOST="0.0.0.0"

# Parámetros de contexto y memoria
CTX_SIZE=$((192*1024))
NGL=99
BATCH_SIZE=1024          # Puedes probar con 2048 si la VRAM lo permite
UBATCH_SIZE=512

# Parámetros de caché KV
CACHE_TYPE_K="q4_0"
CACHE_TYPE_V="q4_0"

# Parámetros de generación para agentes (Recomendados por DeepReinforce)
TEMP=0.6
TOP_P=0.95
TOP_K=20

# Parámetros de rendimiento
THREADS=6
FLASH_ATTN="true"

# Ejecutar llama-server con todas las opciones
llama-server \
    -hf "$MODEL_HF" \
    --port "$PORT" \
    --host "$HOST" \
    -c "$CTX_SIZE" \
    -ngl "$NGL" \
    -b "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top_k "$TOP_K" \
    -t "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --jinja \
    --metrics \
    --repeat-penalty 1.0 \
    --chat-template-file templates/ornith_template.jinja \
    -np 1 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --spec-type draft-mtp --spec-draft-n-max 3
