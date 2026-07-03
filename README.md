# IA — Experimentación con LLMs via llama.cpp

Scripts y herramientas para ejecutar, evaluar y probar modelos de lenguaje grandes (LLMs) usando [`llama.cpp`](https://github.com/ggerganov/llama.cpp) con `llama-server`.

## Estructura

```
.
├── *.sh                    # Scripts de lanzamiento de modelos
├── *.jinja                 # Chat templates personalizados (Jinja2)
└── TEST-IA/
    ├── eval_agentes.py                 # Suite de evaluación (19 tests con informe)
    ├── eval_agentes-gemini2deepseek.py # Evaluación con cliente Gemini→DeepSeek
    ├── comparador.py                   # Comparador de reportes JSON
    ├── test-tools.py                   # Prueba básica de tool calling
    ├── test_tools_extended.py          # Prueba extendida (7 herramientas)
    ├── test_tools_extended_v2.py       # Prueba extendida v2
    ├── llama_bench_complete.py         # Benchmarking de rendimiento
    └── reporte_*.json                  # Reportes de evaluación generados
```

## Modelos disponibles

| Script | Modelo | Especificaciones |
|--------|--------|----------------|
| `Qwythos.sh` | Empero AI Qwythos 9B (MTP) | 131K ctx, speculative decoding, razonamiento |
| `Qwopus36-35B-A3B.sh` | Qwen3.6 35B-A3B (MoE) | 132K ctx, IQ3_XXS, MTP-2, razonamiento |
| `qwen36-35B-A3B.sh` | Qwen3.6 35B-A3B (MoE) | 132K ctx, IQ3_XXS, MTP-2 |
| `qwen36-35B-A3B-AGENTWORLD.sh` | Qwen AgentWorld 35B | 120K ctx, IQ2_M, orientado a agentes |
| `gemma4-26b-a4b-it-qat.sh` | Gemma 4 26B (QAT) | 133K ctx, Q4_K_XL, MTP-4 |
| `gemma4-26b-a4b-it-qat-q4.sh` | Gemma 4 26B (QAT) | Q4_0, cuantización más agresiva |
| `ornith.sh` | Ornith (Mythos 5 1M GGUF) | Variante experimental con Q6_K y Q8_0 |

## Chat templates

Templates Jinja2 personalizados para distintos formatos de tool calling y razonamiento:

- `chat_template.jinja` — Qwythos con tool calls `XML` y bloque `<think>`
- `chat_template2.jinja` — Qwen3.6 v20: detección de errores, truncamiento, multi-step tool calls
- `chat_template-gemini4.jinja` — Formato Gemini-style con `item.properties`
- `chat_template-tollcall.jinja` — Formato toll-call con descripciones inline
- `chat_template-qwythos.jinja` — Variante Qwythos
- `ornith_template.jinja` — Template para Ornith

## Uso básico

```bash
# Iniciar un modelo (ej. Qwythos)
./Qwythos.sh

# El servidor queda escuchando en http://localhost:8080

# Evaluar el modelo con la suite de pruebas
python TEST-IA/eval_agentes.py

# Probar tool calling
python TEST-IA/test-tools.py
```

## Evaluación

`eval_agentes.py` ejecuta 19 pruebas que cubren: coherencia, razonamiento, tool calling, seguridad básica, extracción de datos y más. Genera un reporte JSON con métricas detalladas.

```bash
# Host personalizado y timeout
LLAMA_HOST=http://127.0.0.1:8080 TIMEOUT=180 python TEST-IA/eval_agentes.py
```

## Requisitos

- `llama.cpp` compilado con `llama-server`
- Python 3.14+
- `requests`
