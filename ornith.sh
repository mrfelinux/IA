#!/bin/bash

# Script para iniciar llama-server con Ornith-1.0-9B-GGUF (Q4_K_M)
# Optimizado para 16GB VRAM y uso con Hermes Agent (256K ctx, batch optimizado)

# Configuración del modelo y servidor
#MODEL_HF="deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M"  # Modelo y cuantización[reference:1]
MODEL_HF="deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M"
PORT=8080                                                # Puerto para la API[reference:2]
HOST="0.0.0.0"                                           # Escucha en todas las interfaces

# Parámetros de contexto y memoria
CTX_SIZE=262144                                          # 256K tokens de contexto (nativo del modelo)
NGL=99                                                   # Descargar todas las capas en GPU[reference:5]
BATCH_SIZE=1024                                          # Tamaño de lote para inferencia
UBATCH_SIZE=512                                          # Tamaño de sublote

# Parámetros de caché KV (clave para ahorrar VRAM)
CACHE_TYPE_K="q4_0"                                      # Caché de claves a 4 bits[reference:6]
CACHE_TYPE_V="q4_0"                                      # Caché de valores a 4 bits[reference:7]

# Parámetros de generación para agentes (según benchmarks del modelo)
TEMP=0.6                                                 # Temperatura para coding agents[reference:8]
TOP_P=0.95
TOP_K=20                                               # Nucleus sampling[reference:9]

# Parámetros de rendimiento
THREADS=6                                                # Número de hilos CPU
FLASH_ATTN="true"                                          # Flash Attention para ahorrar memoria[reference:10]

# Ejecutar llama-server con todas las opciones
llama-server \
    -hf "$MODEL_HF" \
    --port "$PORT" \
    --host "$HOST" \
    -c "$CTX_SIZE" \
    -ngl "$NGL" \
    -b "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top_k "$TOP_K" \
    -t "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --metrics \
    --jinja \
    --repeat-penalty 1.0 \
    --chat-template-file ornith_template.jinja \
    -np 1 \
    --metrics
