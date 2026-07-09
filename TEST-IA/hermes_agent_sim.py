#!/usr/bin/env python3
"""
Simulación de test para Hermes-Agent contra llama.cpp server con --metrics.

Simula el bucle ReAct/TAO (Thought → Action → Observation) que usa
Hermes-Agent, incluyendo llamadas a herramientas multi-turno,
tool calls paralelas, y tres escenarios distintos de agente.

Requisitos:
  - llama-server corriendo con --metrics (ej: puerto 8080)
  - Dependencias: pip install requests

Uso:
  python hermes_agent_sim.py                          # localhost:8080
  LLAMA_HOST=http://192.168.1.100:8080 python hermes_agent_sim.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

# ─── Constantes ───────────────────────────────────────────────────────────────

COLOR = os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb" and sys.stdout.isatty()
SERVER = os.getenv("LLAMA_HOST", "http://localhost:8080")
MODEL = "hermes-agent-sim"
DEFAULT_MODEL_ALIAS = "desconocido"

# ─── Colores ANSI ─────────────────────────────────────────────────────────────

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GRAY = "\033[90m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{C_RESET}" if COLOR else text


# ─── Herramientas estilo Hermes-Agent ────────────────────────────────────────

HERMES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Ejecuta un comando en la terminal Bash del sistema. Útil para scripting, manipulación de archivos, y cualquier tarea del sistema operativo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Comando Bash a ejecutar (puede ser multi-línea)."
                    },
                    "description": {
                        "type": "string",
                        "description": "Breve descripción de lo que hace el comando (para logging)."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "TimeOut en segundos para la ejecución (default: 30).",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Lee el contenido de uno o más archivos del sistema de archivos local.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de rutas de archivos a leer."
                    }
                },
                "required": ["files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Escribe contenido en un archivo del sistema. Crea el archivo si no existe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del archivo a escribir."},
                    "content": {"type": "string", "description": "Contenido textual a guardar."},
                    "append": {
                        "type": "boolean",
                        "description": "Si es True, añade al final en vez de sobrescribir.",
                        "default": False
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Realiza una búsqueda en Internet y devuelve resultados relevantes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término de búsqueda."},
                    "max_results": {
                        "type": "integer",
                        "description": "Número máximo de resultados (default: 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Ejecuta código Python arbitrario. Ideal para cálculos, transformación de datos, y scripting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Código Python a ejecutar."},
                    "variables": {
                        "type": "object",
                        "description": "Variables a inyectar en el entorno de ejecución.",
                        "default": {}
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Busca en la memoria persistente del agente por palabras clave. Devuelve fragmentos relevantes de sesiones anteriores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Palabras clave o frase a buscar en memoria."},
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de resultados (default: 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Guarda una observación importante en la memoria persistente del agente para recuperarla en el futuro.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Título corto y descriptivo."},
                    "content": {"type": "string", "description": "Contenido detallado a recordar."},
                    "type": {
                        "type": "string",
                        "enum": ["decision", "fact", "pattern", "preference"],
                        "description": "Tipo de contenido a guardar.",
                        "default": "fact"
                    }
                },
                "required": ["title", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Obtiene el contenido de una URL web (HTML, texto plano, o JSON).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL completa (http/https) a fetchear."},
                    "format": {
                        "type": "string",
                        "enum": ["text", "html", "markdown"],
                        "description": "Formato de salida deseado.",
                        "default": "text"
                    }
                },
                "required": ["url"]
            }
        }
    },
]

# ─── System prompt estilo Hermes-Agent ────────────────────────────────────────

HERMES_SYSTEM_PROMPT = """Eres Hermes-Agent-Sim, un agente autónomo con un bucle Thought-Action-Observation.

Tus habilidades:
- Ejecutar código y comandos (bash, python)
- Leer y escribir archivos (read, write)
- Buscar información en Internet (web_search, web_fetch)
- Gestionar memoria persistente (memory_search, memory_save)

Directrices:
1. PIENSA antes de actuar. Usa herramientas solo cuando sea necesario.
2. Para tareas simples, responde directamente sin llamar herramientas.
3. Cuando necesites información, llama a la herramienta adecuada.
4. Después de recibir el resultado de una herramienta, analízalo antes de continuar.
5. Para tareas complejas, divide el trabajo en pasos y ve ejecutándolos secuencialmente.
6. Si dos herramientas son independientes, puedes llamarlas en paralelo.
7. Temperatura: usa 0.0 para llamadas a herramientas; 0.7 para respuestas finales creativas.
8. Cuando completes la tarea, da una respuesta final clara y concisa al usuario."""


# ─── Escenarios ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ScenarioResult:
    name: str
    turns: list[list[Turn]]
    total_latency: float
    tool_call_count: int
    successful_tool_calls: int
    errors: list[str]
    metrics_snapshot: dict[str, float] = field(default_factory=dict)
    content: str = ""


SCENARIOS: list[dict[str, Any]] = [
    # ── Escenario 1: Multi-herramienta secuencial ──
    {
        "name": "analisis_archivo",
        "desc": "Analiza un archivo Python y genera un resumen (read → python → write)",
        "system": HERMES_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Analiza el archivo test_tools_extended.py del directorio actual. "
                    "Cuenta cuántas funciones define, cuántos casos de prueba hay, "
                    "y escribe un resumen en /tmp/hermes_analisis.txt"
                )
            }
        ],
        "max_turns": 4,
        "min_expected_tool_calls": 2,
        "expected_tools_on_turn": {
            0: ["read", "read_file"],
            1: ["python", "bash"],
            2: ["write", "write_file"],
        },
    },
    # ── Escenario 2: Búsqueda y memoria ──
    {
        "name": "busqueda_memoria",
        "desc": "Busca información y guarda en memoria (web_search → memory_save)",
        "system": (
            HERMES_SYSTEM_PROMPT + (
                "\n\nIMPORTANTE: Tienes acceso a web_search para buscar en Internet. "
                "Cuando encuentres información relevante, guárdala en memoria con memory_save."
            )
        ),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Busca información sobre Python 3.14 features y guarda "
                    "los hallazgos más importantes en tu memoria."
                )
            }
        ],
        "max_turns": 4,
        "min_expected_tool_calls": 2,
        "expected_tools_on_turn": {
            0: ["web_search", "web_fetch"],
            1: ["memory_save"],
        },
    },
    # ── Escenario 3: Código + bash (paralelo posible) ──
    {
        "name": "codigo_y_sistema",
        "desc": "Escribe un script Python, ejecútalo con bash, y captura output",
        "system": HERMES_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Crea un script Python en /tmp/analizador.py que lea este mismo archivo "
                    "(hermes_agent_sim.py), cuente líneas totales, líneas de código, "
                    "líneas de comentarios, y líneas en blanco. Luego ejecútalo y dime los resultados."
                )
            }
        ],
        "max_turns": 5,
        "min_expected_tool_calls": 2,
        "expected_tools_on_turn": {
            0: ["write", "write_file"],
            1: ["bash"],
        },
    },
    # ── Escenario 4: Consulta directa (sin herramientas) ──
    {
        "name": "consulta_directa",
        "desc": "Pregunta simple que NO requiere tool calling (evalúa que el modelo no invoque herramientas innecesariamente)",
        "system": HERMES_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": "¿Cuál es la capital de Japón y cuál es su población aproximada?"
            }
        ],
        "max_turns": 1,
        "min_expected_tool_calls": 0,
        "no_tool_call_expected": True,
    },
    # ── Escenario 5: Pipeline complejo (bash → python → read → write) ──
    {
        "name": "pipeline_datos",
        "desc": "Pipeline de datos: genera datos, procésalos, guarda resultados",
        "system": HERMES_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ejecuta 'ls -la *.py' en el directorio actual para listar los scripts Python. "
                    "Luego, para cada archivo, cuenta cuántas líneas tiene usando Python. "
                    "Finalmente, escribe un archivo CSV en /tmp/hermes_report.csv con "
                    "los resultados: nombre_archivo, lineas, tamano_bytes."
                )
            }
        ],
        "max_turns": 6,
        "min_expected_tool_calls": 3,
        "expected_tools_on_turn": {
            0: ["bash"],
            1: ["python", "bash"],
            2: ["write", "write_file"],
        },
    },
]


# ─── Funciones de utilidad para el servidor ───────────────────────────────────

def root_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    if url.endswith("/v1/"):
        return url[:-4]
    return url


def server_health(server_url: str) -> bool:
    try:
        r = requests.get(f"{root_url(server_url)}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def get_metrics(server_url: str) -> str | None:
    try:
        r = requests.get(f"{root_url(server_url)}/metrics", timeout=5)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def get_props(server_url: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{root_url(server_url)}/props", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_model_name(server_url: str) -> str:
    props = get_props(server_url)
    if props:
        alias = props.get("model_alias") or props.get("model_path", "")
        if alias:
            import re
            alias = re.sub(r'[^\w.-]', '_', alias).strip('_.') or DEFAULT_MODEL_ALIAS
            return alias.lower().replace(' ', '_')
    return DEFAULT_MODEL_ALIAS


def parse_metrics(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].split("{")[0].replace(":", "_")
                parsed[key] = float(parts[1])
        except ValueError:
            pass
    return parsed


def metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    delta: dict[str, float] = {}
    for k in keys:
        b = before.get(k, 0.0)
        a = after.get(k, 0.0)
        if a != b:
            delta[k] = a - b
    return delta


# ─── Mock de ejecución de herramientas ────────────────────────────────────────

def _execute_tool_mock(name: str, args: dict[str, Any]) -> str:
    """Simula la ejecución de herramientas. Devuelve resultado ficticio."""
    match name:
        case "bash":
            cmd = args.get("command", "")
            if "ls " in cmd:
                return "hermes_agent_sim.py\ntest_tools_extended.py\ntools_defs.py\nllama_bench_complete.py\ncomparador.py\nia-test.py"
            if "wc -l" in cmd or "count" in cmd.lower():
                return "120 hermes_agent_sim.py"
            return f"[bash OK] Comando ejecutado: {cmd[:60]}..."
        case "read" | "read_file":
            files = args.get("files", [args.get("path", "/tmp/default")])
            return "\n---\n".join(
                f"=== {f} ===\n(contenido simulado de {f})" for f in files
            )
        case "write" | "write_file":
            path = args.get("path", "/tmp/output.txt")
            return f"[OK] {len(args.get('content', ''))} bytes escritos en {path}"
        case "python":
            code = args.get("code", "")
            if "len(" in code or "count" in code:
                return "42"
            if "print" in code:
                return "Hello from Python mock"
            return f"[Python mock] Ejecutado ({len(code)} chars)"
        case "web_search":
            query = args.get("query", "")
            return (
                f"Resultados de búsqueda para '{query}':\n"
                f"1. Python 3.14 introduces the Global Interpreter Lock (GIL) removal PEP 703\n"
                f"2. New pattern matching features and improved error messages\n"
                f"3. Performance improvements: 15% faster CPython interpreter"
            )
        case "web_fetch":
            url = args.get("url", "")
            return f"[Contenido de {url}] (simulado) - Página obtenida correctamente, 12500 bytes."
        case "memory_search":
            q = args.get("query", "")
            return f"[Memoria] No se encontraron resultados previos para '{q}'."
        case "memory_save":
            return f"[Memoria] Guardado: '{args.get('title', 'sin título')}'"
        case _:
            return f"[{name}] Ejecutado con args: {json.dumps(args)}"


# ─── Llamada a la API ────────────────────────────────────────────────────────

def chat_completion(
    server_url: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    stream: bool = False,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        resp = requests.post(
            f"{server_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


# ─── Simulación del bucle ReAct ───────────────────────────────────────────────

def simulate_agent_loop(
    server_url: str,
    scenario: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> ScenarioResult:
    name = scenario["name"]
    max_turns = scenario["max_turns"]
    min_tool_calls = scenario.get("min_expected_tool_calls", 0)

    messages: list[dict[str, Any]] = []
    if scenario.get("system"):
        messages.append({"role": "system", "content": scenario["system"]})
    messages.extend(scenario["messages"])

    turns_log: list[list[Turn]] = []
    tool_call_count = 0
    successful_tool_calls = 0
    errors: list[str] = []
    final_content = ""
    start_time = time.time()

    for turn_idx in range(max_turns):
        turn_tools: list[Turn] = []
        prompt_tokens_before = len(json.dumps(messages))

        response = chat_completion(server_url, messages, tools=tools)

        if response is None:
            err = f"Turno {turn_idx}: Sin respuesta del servidor"
            errors.append(err)
            turn_tools.append(Turn(role="error", content=err))
            turns_log.append(turn_tools)
            break

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "")

        content = message.get("content", "") or ""
        if content:
            turn_tools.append(Turn(role="assistant", content=content))
            final_content = content

        # tool_calls nativos (OpenAI format)
        tool_calls = message.get("tool_calls", [])
        if finish == "tool_calls" or tool_calls:
            tool_call_count += len(tool_calls)
            turn_tc: list[dict[str, Any]] = []

            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "?")
                try:
                    func_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    func_args = {}

                turn_tc.append({
                    "id": tc.get("id", ""),
                    "name": func_name,
                    "args": func_args,
                })

                # Simular ejecución
                result = _execute_tool_mock(func_name, func_args)
                turn_tools.append(Turn(
                    role="tool",
                    content=result,
                    tool_call_id=tc.get("id", ""),
                    name=func_name,
                ))
                successful_tool_calls += 1

            turns_log.append(turn_tools)

            # Preparar siguiente turno
            messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "?")
                try:
                    func_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    func_args = {}
                result = _execute_tool_mock(func_name, func_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

            continue  # siguiente turno

        # Sin tool_calls → respuesta final
        if content:
            turn_tools.append(Turn(role="assistant", content=content))
            final_content = content
        turns_log.append(turn_tools)
        break  # fin del bucle

    total_latency = time.time() - start_time

    return ScenarioResult(
        name=name,
        turns=turns_log,
        total_latency=total_latency,
        tool_call_count=tool_call_count,
        successful_tool_calls=successful_tool_calls,
        errors=errors,
        content=final_content,
    )


# ─── Evaluación de resultados ────────────────────────────────────────────────

def evaluate_scenario(scenario: dict[str, Any], result: ScenarioResult) -> tuple[bool, str]:
    issues: list[str] = []

    min_tool_calls = scenario.get("min_expected_tool_calls", 0)
    if result.tool_call_count < min_tool_calls:
        issues.append(
            f"Tool calls esperadas ≥{min_tool_calls}, obtenidas {result.tool_call_count}"
        )

    if scenario.get("no_tool_call_expected") and result.tool_call_count > 0:
        issues.append(
            f"No se esperaban tool calls, se obtuvieron {result.tool_call_count}"
        )

    if result.errors:
        issues.append(f"Errores: {'; '.join(result.errors[:3])}")

    if result.total_latency > 120:
        issues.append(f"Latencia excesiva: {result.total_latency:.1f}s > 120s")

    passed = len(issues) == 0
    summary = "OK" if passed else issues[0]
    return passed, summary


# ─── Reporte ──────────────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{_c(C_BOLD + C_CYAN, '=' * 72)}")
    print(f"{_c(C_BOLD + C_CYAN, f'  {text}')}")
    print(f"{_c(C_BOLD + C_CYAN, '=' * 72)}")


def print_result(name: str, passed: bool, summary: str, latency: float, tools: int):
    icon = _c(C_GREEN, "✅") if passed else _c(C_RED, "❌")
    lat_str = f"{latency:.2f}s" if latency < 60 else f"{latency/60:.1f}min"
    print(f"  {icon} {_c(C_BOLD, name):<30} {tools} tools | {lat_str} | {summary}")


def show_report(
    scenario_results: list[tuple[dict[str, Any], ScenarioResult]],
    global_metrics: dict[str, float],
):
    print_header("📊 INFORME DE SIMULACIÓN HERMES-AGENT")

    passed = sum(1 for _, r in scenario_results if not r.errors)
    total = len(scenario_results)

    # Tabla de escenarios
    print(f"\n{_c(C_BOLD, 'Escenario'):<35} {'Tools':<8} {'Latencia':<12} {'Resultado'}")
    print("-" * 72)

    for scenario, result in scenario_results:
        ok, summary = evaluate_scenario(scenario, result)
        icon = _c(C_GREEN, "✅") if ok else _c(C_RED, "❌")
        lat = f"{result.total_latency:.2f}s"
        print(f"  {icon} {scenario['name']:<30} {result.tool_call_count:<8} {lat:<12} {summary}")

    print(f"\n{_c(C_BOLD, '📍 Resumen de escenarios:')} {passed}/{total} exitosos")

    # Métricas globales
    if global_metrics:
        print(f"\n{_c(C_BOLD, '⚡ Métricas del servidor (Prometheus)')}")
        metric_labels = {
            "llamacpp_prompt_tokens_seconds": "Velocidad eval prompt (tok/s)",
            "llamacpp_predicted_tokens_seconds": "Velocidad generación (tok/s)",
            "llamacpp_kv_cache_usage_ratio": "Uso KV Cache (ratio)",
            "llamacpp_requests_processing": "Solicitudes activas",
            "llamacpp_tokens_predicted_total": "Tokens generados total",
            "llamacpp_prompt_tokens_total": "Tokens prompt total",
        }
        for key, desc in metric_labels.items():
            if key in global_metrics:
                val = global_metrics[key]
                print(f"  • {desc:<35}: {val:.4f}" if val < 1 else f"  • {desc:<35}: {val:.2f}")

    # Velocidad promedio
    total_tokens = sum(
        r.tool_call_count + bool(r.content)
        for _, r in scenario_results
    )
    total_time = sum(r.total_latency for _, r in scenario_results)
    if total_time > 0:
        print(f"  • {'Throughput general (pasos/s)':<35}: {total_tokens / total_time:.2f}")

    print(f"\n{_c(C_GREEN, '✅ Simulación completada.')}")


# ─── Exportación JSON ─────────────────────────────────────────────────────────

ScenarioResultDict = dict[str, Any]

def _result_to_dict(scenario: dict[str, Any], result: ScenarioResult) -> ScenarioResultDict:
    ok, msg = evaluate_scenario(scenario, result)
    return {
        "name": result.name,
        "desc": scenario.get("desc", ""),
        "passed": ok,
        "summary": msg,
        "total_latency_s": round(result.total_latency, 3),
        "tool_call_count": result.tool_call_count,
        "successful_tool_calls": result.successful_tool_calls,
        "errors": result.errors,
        "final_content_preview": result.content[:200] if result.content else "",
        "turns_count": len(result.turns),
    }


def save_results_to_json(
    server_url: str,
    scenario_results: list[tuple[dict[str, Any], ScenarioResult]],
    global_metrics: dict[str, float],
    model_alias: str,
    output_path: str | None = None,
) -> str:
    timestamp = datetime.now(UTC)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

    if not output_path:
        safe_name = model_alias.replace(" ", "_").replace("/", "-")
        output_path = f"hermes-{safe_name}.{ts_str}.json"

    scenarios_json = [_result_to_dict(s, r) for s, r in scenario_results]
    passed = sum(1 for s in scenarios_json if s["passed"])
    total = len(scenarios_json)

    data: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "model": model_alias,
        "server_url": server_url,
        "resumen": {
            "escenarios_exitosos": passed,
            "escenarios_totales": total,
            "porcentaje_exito": f"{passed / total * 100:.1f}%" if total else "0%",
            "latencia_total_s": round(sum(s["total_latency_s"] for s in scenarios_json), 3),
            "tool_calls_totales": sum(s["tool_call_count"] for s in scenarios_json),
        },
        "metricas_servidor": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in global_metrics.items()
        },
        "escenarios": scenarios_json,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n{_c(C_GREEN, '💾')} Resultados guardados en: {_c(C_BOLD, output_path)}")
    except Exception as e:
        print(f"\n{_c(C_RED, '❌')} Error al guardar JSON: {e}")

    return output_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulación de test Hermes-Agent contra llama.cpp con --metrics"
    )
    parser.add_argument(
        "-u", "--url",
        default=SERVER,
        help=f"URL del servidor (default: {SERVER})",
    )
    parser.add_argument(
        "--scenario",
        help="Ejecutar solo un escenario por nombre (ej: 'analisis_archivo')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar escenarios y salir (sin contacto con servidor)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Ruta del JSON de salida (default: hermes-<modelo>.<timestamp>.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar resultados en JSON",
    )

    args = parser.parse_args()
    server = args.url.rstrip("/")

    if not args.dry_run:
        print(f"{_c(C_CYAN, '🔍')} Verificando servidor en {server}...")
        if not server_health(server):
            print(_c(C_RED, f"❌ No hay conexión con {server}/health"))
            print("   Asegúrate de que llama-server esté corriendo con --metrics")
            sys.exit(1)
        print(_c(C_GREEN, "   ✅ Servidor OK"))

    model_alias = get_model_name(server) if not args.dry_run else DEFAULT_MODEL_ALIAS
    print(f"  {_c(C_GRAY, 'Modelo detectado:')} {_c(C_BOLD, model_alias)}")

    # Filtrar escenarios
    scenarios = [
        s for s in SCENARIOS
        if not args.scenario or args.scenario in s["name"]
    ]
    if not scenarios:
        print(_c(C_YELLOW, f"⚠️  No se encontró escenario '{args.scenario}'"))
        names = [s["name"] for s in SCENARIOS]
        print(f"   Escenarios disponibles: {', '.join(names)}")
        sys.exit(1)

    print_header(f"🚀 SIMULACIÓN HERMES-AGENT ({len(scenarios)} escenarios)")

    if args.dry_run:
        for s in scenarios:
            print(f"  • {_c(C_BOLD, s['name']):<30} — {s['desc']}")
            print(f"    Turns máx: {s['max_turns']}, Tools min esperadas: {s.get('min_expected_tool_calls', 0)}")
        print(f"\n{_c(C_GREEN, '✅ Dry-run completado.')}")
        return

    metrics_before = parse_metrics(get_metrics(server))
    scenario_results: list[tuple[dict[str, Any], ScenarioResult]] = []

    for scenario in scenarios:
        print(f"\n  {_c(C_BOLD, '▶')} Escenario: {_c(C_BOLD, scenario['name'])}")
        print(f"    {scenario['desc']}")

        t0 = time.time()
        result = simulate_agent_loop(server, scenario, tools=HERMES_TOOLS)
        elapsed = time.time() - t0

        ok, summary = evaluate_scenario(scenario, result)
        print_result(scenario["name"], ok, summary, elapsed, result.tool_call_count)

        for err in result.errors[:2]:
            print(f"    {_c(C_RED, '⚠')} {err}")
        scenario_results.append((scenario, result))

    metrics_after = parse_metrics(get_metrics(server))
    global_delta = metrics_delta(metrics_before, metrics_after)

    show_report(scenario_results, global_delta)

    if not args.no_save:
        save_results_to_json(server, scenario_results, global_delta, model_alias, args.output)


if __name__ == "__main__":
    main()
