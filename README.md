# IA — Experimentación con LLMs via llama.cpp

Scripts y herramientas para ejecutar, evaluar y probar modelos de lenguaje grandes (LLMs) usando [`llama.cpp`](https://github.com/ggerganov/llama.cpp) con `llama-server`.

## Estructura

```
.
├── *.sh                         # Scripts de lanzamiento de modelos (13)
├── templates/
│   └── *.jinja                  # Chat templates personalizados (Jinja2, 6)
├── .env.example                 # Ejemplo de configuración de entorno
├── .gga                         # Config de code review (Gentle Guardian Angel)
├── pyproject.toml               # Dependencias Python
├── AGENTS.md                    # Reglas para asistentes AI
├── TEST-IA/
│   ├── ia-test.py               # Suite de evaluación (19 tests con informe)
│   ├── comparador.py            # Comparador de reportes JSON
│   ├── llama_bench_complete.py  # Benchmarking de rendimiento
│   ├── test-tools.py            # Prueba básica de tool calling
│   ├── test_tools_extended.py   # Prueba extendida (7 herramientas)
│   ├── test_tools_extended_v2.py# Prueba extendida v2
│   ├── tools_defs.py            # Definiciones compartidas de herramientas
│   ├── tests/                   # Tests unitarios (pytest)
│   └── reporte_*.json           # Reportes de evaluación generados
```

## Modelos disponibles

| Script | Modelo | Especificaciones |
|--------|--------|----------------|
| `Qwythos.sh` | Empero AI Qwythos 9B (MTP) | 128K ctx, Q8_0, speculative decoding, razonamiento |
| `Qwopus.sh` | Jackrong Qwopus 3.5 9B (MTP) | 128K ctx, Q8_0, MTP-2, razonamiento |
| `qwen36-27B-MTP.sh` | Qwen3.6 27B (MTP) | 128K ctx, IQ3_XXS, MTP-4 |
| `qwen36-35B-A3B.sh` | Qwen3.6 35B-A3B (MoE) | 128K ctx, IQ3_XXS, MTP-2 |
| `qwen36-35B-A3B-AGENTWORLD.sh` | Qwen AgentWorld 35B | 120K ctx, IQ2_M, orientado a agentes |
| `qwen36-35B-A3B-TEST.sh` | Qwen3.6 35B-A3B (TEST) | 128K ctx, IQ3_XXS, YaRN rope scaling |
| `gemma4-12b-it.sh` | Gemma 4 12B IT | 128K ctx, Q4_K_XL, MTP-4, razonamiento |
| `gemma4-12b-it-2.sh` | Gemma 4 12B IT (QAT) | 128K ctx, Q4_K_XL, MTP-4 |
| `gemma4-12b-agentic.sh` | Gemma 4 12B Agentic (R1) | 128K ctx, Q6_K, MTP-4, razonamiento |
| `gemma4-26b-a4b-it-qat.sh` | Gemma 4 26B (QAT) | 128K ctx, Q4_K_XL, MTP-4 |
| `gemma4-26b-a4b-it-qat-q4.sh` | Gemma 4 26B (QAT) | Q4_0, cuantización más agresiva |
| `ornith.sh` | Ornith (Mythos 5 1M GGUF) | Variante experimental Q6_K y Q8_0 |
| `ornith-mtp.sh` | Ornith 1.0 9B (MTP) | 128K ctx, Q8_0, MTP-3, chat template ornith |

## Chat templates

Templates Jinja2 personalizados para distintos formatos de tool calling y razonamiento, ubicados en `templates/`:

| Archivo | Descripción |
|---------|-------------|
| `chat_template.jinja` | Qwythos con tool calls XML y bloque `<think>` |
| `chat_template2.jinja` | Qwen3.6 v20: detección de errores, truncamiento, multi-step |
| `chat_template-gemini4.jinja` | Formato Gemini-style con `item.properties` |
| `chat_template-tollcall.jinja` | Formato toll-call con descripciones inline |
| `chat_template-qwythos.jinja` | Variante Qwythos |
| `ornith_template.jinja` | Template para Ornith |

## Configuración

Las siguientes variables de entorno modifican el comportamiento de la suite de pruebas (`ia-test.py`):

| Variable | Valor por defecto | Descripción |
|----------|------------------|-------------|
| `LLAMA_HOST` | `http://127.0.0.1:8080` | URL base del servidor `llama-server` |
| `TIMEOUT` | `120` | Timeout en segundos por petición |
| `MAX_TOKENS` | `4096` | Máximo de tokens generados por respuesta |
| `QUIET` | `0` | Suprimir salida detallada (`1`, `true` o `yes`) |
| `LOG_FILE` | _(ninguno)_ | Ruta para guardar log de evaluación |
| `TESTS_FILTER` | _(ninguno)_ | Filtrar tests por nombre (subcadena) |

```bash
LLAMA_HOST=http://192.168.1.10:8080 TIMEOUT=300 QUIET=1 python TEST-IA/ia-test.py
```

## Uso básico

```bash
# Iniciar un modelo (ej. Qwythos)
./Qwythos.sh

# El servidor queda escuchando en http://localhost:8080

# Evaluar el modelo con la suite de pruebas
python TEST-IA/ia-test.py

# Probar tool calling
python TEST-IA/test-tools.py
```

## Evaluación

`ia-test.py` ejecuta 19 pruebas que cubren: coherencia, razonamiento, tool calling, seguridad básica, extracción de datos y más. Genera un reporte JSON con métricas detalladas.

```bash
# Host personalizado y timeout
LLAMA_HOST=http://127.0.0.1:8080 TIMEOUT=180 python TEST-IA/ia-test.py

# Ejecutar solo tests específicos
TESTS_FILTER=tool python TEST-IA/ia-test.py
```

## Desarrollo

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e .
```

Las dependencias se definen en `pyproject.toml`. Actualmente solo requiere `requests`.

## Requisitos

- `llama.cpp` compilado con `llama-server`
- Python 3.14+
- `requests`
- **Hardware** — Mínimo: depende del modelo (ver tabla). Recomendado: 16 GB VRAM para modelos de 9B–12B en Q4 o superior.
