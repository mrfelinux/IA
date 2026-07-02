#!/usr/bin/env python3
"""
Prueba extendida y avanzada de tool calling con llama.cpp.
Evalúa la capacidad de selección de herramientas, extracción de parámetros,
casos ambiguos, consultas conversacionales y llamadas secuenciales.

Python 3.14+ requerido.
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Never, Self

import requests

# ─── Type aliases (PEP 695 — Python 3.12+) ──────────────────────────────────

type ToolDef = dict[str, Any]
type TestCase = dict[str, Any]
type Message = dict[str, str]
type ArgsSubset = dict[str, str | int | float | bool]
type EvalResult = tuple[bool, str, str | None, dict[str, Any]]
type ToolCallParsed = tuple[str | None, dict[str, Any], str | None]


# ─── Configuración (frozen dataclass con slots — Python 3.10+) ──────────────

@dataclass(slots=True, frozen=True)
class Config:
    server: str = "http://localhost:8080/v1"
    model: str = "cualquiera"
    timeout: int = 60
    verbose: bool = False
    color: bool = True
    output: Path | None = None
    tests: frozenset[int] | None = None
    repeat: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0


# ─── Colores ANSI ─────────────────────────────────────────────────────────────

class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    DIM = "\033[2m"

    @classmethod
    def disable(cls) -> None:
        for attr in ("RESET", "BOLD", "RED", "GREEN", "YELLOW", "BLUE", "CYAN", "DIM"):
            setattr(cls, attr, "")


def cprint(color: str, text: str, **kwargs: Any) -> None:
    print(f"{color}{text}{Color.RESET}", **kwargs)


def verbose_print(cfg: Config, text: str) -> None:
    if cfg.verbose:
        cprint(Color.DIM, f"  [DEBUG] {text}")


# ─── Definición de herramientas (14) ─────────────────────────────────────────

TOOLS: list[ToolDef] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtiene el clima actual de una ciudad",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nombre de la ciudad"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca información general en Internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término de búsqueda"},
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Número de resultados",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Envía un correo electrónico a un destinatario",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Dirección de correo del destinatario",
                    },
                    "subject": {"type": "string", "description": "Asunto del mensaje"},
                    "body": {"type": "string", "description": "Cuerpo del mensaje"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Ejecuta un fragmento de código Python y devuelve el resultado",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Código Python a ejecutar",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un archivo local del sistema de archivos",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta absoluta o relativa del archivo",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe o guarda contenido textual en un archivo del sistema",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del archivo a escribir"},
                    "content": {
                        "type": "string",
                        "description": "Texto o contenido a guardar",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Ejecuta una consulta SQL en la base de datos de la aplicación",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Sentencia SQL a ejecutar",
                    },
                    "database": {
                        "type": "string",
                        "default": "default",
                        "description": "Nombre de la base de datos",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Realiza cálculos matemáticos o evalúa expresiones aritméticas complejas",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expresión matemática (ej. '45 * 12 + sqrt(144)')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Traduce un texto de un idioma de origen a un idioma de destino",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texto original a traducir",
                    },
                    "target_lang": {
                        "type": "string",
                        "description": "Idioma destino (ej. 'inglés', 'francés', 'en', 'fr')",
                    },
                    "source_lang": {
                        "type": "string",
                        "description": "Idioma origen opcional",
                    },
                },
                "required": ["text", "target_lang"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Obtiene la cotización bursátil o precio actual de las acciones de una empresa",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Símbolo o ticker de la acción (ej. AAPL, GOOGL, TSLA)",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Agenda un evento o reunión en el calendario",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Título o asunto del evento",
                    },
                    "date_time": {
                        "type": "string",
                        "description": "Fecha y hora del evento (ej. '2026-07-01 10:00')",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "default": 60,
                        "description": "Duración en minutos",
                    },
                },
                "required": ["title", "date_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_web_page",
            "description": "Obtiene el contenido HTML o texto de una URL web específica",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL completa de la página web (http/https)",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_command",
            "description": "Ejecuta un comando en la terminal Bash del sistema operativo",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando Bash a ejecutar"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Genera una imagen digital a partir de una descripción en texto (prompt)",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Descripción detallada de la imagen a generar",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "default": "1:1",
                        "description": "Relación de aspecto (ej. 1:1, 16:9)",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


# ─── Casos de prueba ─────────────────────────────────────────────────────────

TEST_CASES: list[TestCase] = [
    # --- Herramientas básicas ---
    {
        "id": 1,
        "desc": "Clima en Madrid",
        "category": "Básico",
        "messages": [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}],
        "expected_tool": "get_weather",
        "expected_args_subset": {"city": "Madrid"},
    },
    {
        "id": 2,
        "desc": "Búsqueda web sobre fotosíntesis",
        "category": "Básico",
        "messages": [
            {"role": "user", "content": "Busca información sobre la fotosíntesis"}
        ],
        "expected_tool": "search_web",
        "expected_args_subset": {"query": "fotosíntesis"},
    },
    {
        "id": 3,
        "desc": "Enviar correo electrónico",
        "category": "Básico",
        "messages": [
            {
                "role": "user",
                "content": "Envía un email a juan@example.com con asunto 'Hola' y cuerpo 'Saludos cordiales'",
            }
        ],
        "expected_tool": "send_email",
        "expected_args_subset": {
            "to": "juan@example.com",
            "subject": "Hola",
            "body": "Saludos cordiales",
        },
    },
    {
        "id": 4,
        "desc": "Ejecutar código Python",
        "category": "Básico",
        "messages": [
            {"role": "user", "content": "Ejecuta este código: print('Hola mundo')"}
        ],
        "expected_tool": "run_python_code",
        "expected_args_subset": {"code": "print('Hola mundo')"},
    },
    {
        "id": 5,
        "desc": "Leer archivo local",
        "category": "Básico",
        "messages": [
            {"role": "user", "content": "Muéstrame el contenido de /etc/hosts"}
        ],
        "expected_tool": "read_file",
        "expected_args_subset": {"path": "/etc/hosts"},
    },
    {
        "id": 6,
        "desc": "Escribir archivo local",
        "category": "Básico",
        "messages": [
            {"role": "user", "content": "Guarda 'Hola mundo' en /tmp/saludo.txt"}
        ],
        "expected_tool": "write_file",
        "expected_args_subset": {"path": "/tmp/saludo.txt", "content": "Hola mundo"},
    },
    {
        "id": 7,
        "desc": "Consulta SQL",
        "category": "Básico",
        "messages": [
            {"role": "user", "content": "Ejecuta esta consulta: SELECT * FROM usuarios"}
        ],
        "expected_tool": "query_database",
        "expected_args_subset": {"query": "SELECT * FROM usuarios"},
    },
    # --- Herramientas adicionales ---
    {
        "id": 8,
        "desc": "Cálculo matemático",
        "category": "Matemáticas",
        "messages": [
            {"role": "user", "content": "¿Cuánto es 35 multiplicado por 14 sumado a 120?"}
        ],
        "expected_tool": "calculate",
        "expected_args_subset": {},
    },
    {
        "id": 9,
        "desc": "Traducción de texto",
        "category": "Idiomas",
        "messages": [
            {
                "role": "user",
                "content": "Traduce al inglés la frase 'El conocimiento es poder'",
            }
        ],
        "expected_tool": "translate_text",
        "expected_args_subset": {"text": "El conocimiento es poder"},
    },
    {
        "id": 10,
        "desc": "Cotización bursátil",
        "category": "Finanzas",
        "messages": [
            {
                "role": "user",
                "content": "¿Cuál es el precio actual de las acciones de Apple (AAPL)?",
            }
        ],
        "expected_tool": "get_stock_price",
        "expected_args_subset": {"ticker": "AAPL"},
    },
    {
        "id": 11,
        "desc": "Agendar evento en calendario",
        "category": "Agenda",
        "messages": [
            {
                "role": "user",
                "content": "Agenda una reunión llamada 'Revisión de Proyecto' para mañana a las 10:00",
            }
        ],
        "expected_tool": "create_calendar_event",
        "expected_args_subset": {"title": "Revisión de Proyecto"},
    },
    {
        "id": 12,
        "desc": "Navegar URL específica",
        "category": "Web",
        "messages": [
            {
                "role": "user",
                "content": "Descarga o lee la página web https://example.com/api",
            }
        ],
        "expected_tool": "fetch_web_page",
        "expected_args_subset": {"url": "https://example.com/api"},
    },
    {
        "id": 13,
        "desc": "Comando Bash",
        "category": "Sistema",
        "messages": [
            {
                "role": "user",
                "content": "Ejecuta el comando bash `ls -la /var/log` para ver los archivos",
            }
        ],
        "expected_tool": "system_command",
        "expected_args_subset": {"command": "ls -la /var/log"},
    },
    {
        "id": 14,
        "desc": "Generación de imagen",
        "category": "Multimedia",
        "messages": [
            {
                "role": "user",
                "content": "Genera una imagen de un gato futurista con gafas de sol en un estilo cyberpunk",
            }
        ],
        "expected_tool": "generate_image",
        "expected_args_subset": {},
    },
    # --- Casos complejos / Ambigüedad / Sin Herramienta ---
    {
        "id": 15,
        "desc": "Ambigüedad clima vs búsqueda",
        "category": "Ambigüedad",
        "messages": [
            {"role": "user", "content": "Busca en Internet el clima de París"}
        ],
        "expected_tool": "get_weather",
        "expected_args_subset": {"city": "París"},
    },
    {
        "id": 16,
        "desc": "Charla conversacional (Sin Herramienta)",
        "category": "Conversacional",
        "messages": [
            {"role": "user", "content": "Hola, ¿cómo estás hoy y de qué eres capaz?"}
        ],
        "expected_tool": None,
        "expected_args_subset": {},
    },
    # --- Multi-tool: llamadas secuenciales ---
    {
        "id": 17,
        "desc": "Clima + email condicional",
        "category": "Multi-tool",
        "messages": [
            {
                "role": "user",
                "content": "¿Qué tiempo hace en Madrid? Si hace sol, envía un email a ana@test.com con asunto 'Día soleado'",
            }
        ],
        "sequential": True,
        "expected_tool_sequence": [
            {"tool": "get_weather", "args_subset": {"city": "Madrid"}},
            {"tool": "send_email", "args_subset": {"to": "ana@test.com"}},
        ],
    },
    {
        "id": 18,
        "desc": "Búsqueda + ejecución de código",
        "category": "Multi-tool",
        "messages": [
            {
                "role": "user",
                "content": "Busca la fórmula del área del círculo y luego ejecuta un código Python que la calcule con radio 5",
            }
        ],
        "sequential": True,
        "expected_tool_sequence": [
            {"tool": "search_web", "args_subset": {"query": "área del círculo"}},
            {"tool": "run_python_code", "args_subset": {"code": "5"}},
        ],
    },
]


# ─── Llamada HTTP con retry ──────────────────────────────────────────────────

def chat_completion(
    cfg: Config,
    messages: list[Message],
    tools: list[ToolDef],
    tool_choice: str = "auto",
    temperature: float = 0.0,
) -> tuple[dict[str, Any] | None, float, str | None]:
    """Realiza una petición chat/completions con reintentos."""
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
    }

    last_error: str | None = None
    total_elapsed = 0.0
    resp: requests.Response | None = None

    for attempt in range(1, cfg.max_retries + 1):
        start = time.monotonic()
        try:
            resp = requests.post(
                f"{cfg.server}/chat/completions",
                json=payload,
                timeout=cfg.timeout,
            )
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            resp.raise_for_status()
            return resp.json(), total_elapsed, None

        except requests.exceptions.ConnectionError:
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            last_error = (
                f"No se pudo conectar a {cfg.server}. "
                f"¿Está llama-server activo? (intento {attempt}/{cfg.max_retries})"
            )

        except requests.exceptions.Timeout:
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            last_error = (
                f"Timeout después de {cfg.timeout}s "
                f"(intento {attempt}/{cfg.max_retries})"
            )

        except requests.exceptions.HTTPError:
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            status = resp.status_code if resp is not None else "N/A"
            body = resp.text[:200] if resp is not None else "Sin body"
            last_error = (
                f"HTTP {status}: {body} "
                f"(intento {attempt}/{cfg.max_retries})"
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            last_error = (
                f"Error inesperado: {e} (intento {attempt}/{cfg.max_retries})"
            )

        # Backoff exponencial
        if attempt < cfg.max_retries:
            delay = cfg.retry_delay * (2 ** (attempt - 1))
            verbose_print(cfg, f"Esperando {delay:.1f}s antes del siguiente intento...")
            time.sleep(delay)

    return None, total_elapsed, last_error


# ─── Parsing de tool_calls ───────────────────────────────────────────────────

def _parse_tool_call(call: dict[str, Any]) -> ToolCallParsed:
    """Extrae nombre y argumentos parseados de un tool_call."""
    func: dict[str, Any] = call.get("function", {})
    name: str | None = func.get("name")
    args_str: str | dict = func.get("arguments", "{}")

    if isinstance(args_str, dict):
        return name, args_str, None

    try:
        return name, json.loads(args_str), None
    except json.JSONDecodeError as jde:
        return name, {}, f"JSON inválido: {jde}"


# ─── Verificación de argumentos ──────────────────────────────────────────────

def _check_args(
    tool_name: str,
    actual_args: dict[str, Any],
    expected_subset: ArgsSubset,
    step: int | None = None,
) -> str | None:
    """Verifica subconjunto de argumentos. Devuelve hint si falla, None si ok."""
    if not expected_subset:
        return None

    step_prefix = f"Paso {step}: " if step else ""

    for key, expected_val in expected_subset.items():
        if key not in actual_args:
            return (
                f"{step_prefix}PISTA: La herramienta '{tool_name}' falta el "
                f"parámetro '{key}'.\n   Parámetros extraídos: {list(actual_args.keys())}."
            )

        actual = actual_args[key]

        # match para diferenciar tipos de comparación
        match expected_val:
            case str():
                if expected_val.lower() not in str(actual).lower():
                    return (
                        f"{step_prefix}PISTA: '{key}' = '{actual}', "
                        f"no contiene '{expected_val}'."
                    )
            case _ if actual != expected_val:
                return (
                    f"{step_prefix}PISTA: '{key}' = {actual} ({type(actual).__name__}), "
                    f"se esperaba {expected_val}."
                )

    return None


# ─── Evaluación principal ────────────────────────────────────────────────────

def evaluate(
    case: TestCase, response: dict[str, Any] | None, err_msg: str | None
) -> EvalResult:
    """
    Evalúa la respuesta del modelo.
    Devuelve (exito, mensaje, pista, detalles).
    """
    details: dict[str, Any] = {
        "raw_content": None,
        "tool_calls_raw": [],
        "parsed_args": {},
        "tool_name": None,
        "finish_reason": None,
    }

    if not response:
        hint = (
            f"PISTA: {err_msg or 'Sin respuesta del servidor.'}\n"
            f"   Verifica que llama-server esté ejecutándose en el servidor configurado."
        )
        return False, "Error de conexión o HTTP", hint, details

    choice: dict[str, Any] = response.get("choices", [{}])[0]
    finish: str | None = choice.get("finish_reason")
    message: dict[str, Any] = choice.get("message", {})
    content: str | None = message.get("content", "")
    tool_calls: list[dict[str, Any]] = message.get("tool_calls", [])

    details["finish_reason"] = finish
    details["raw_content"] = content
    details["tool_calls_raw"] = tool_calls

    # ── Multi-tool: secuencia ──
    if case.get("sequential"):
        return _evaluate_sequential(case, tool_calls, finish, details)

    expected_tool: str | None = case.get("expected_tool")

    # match para los diferentes escenarios de evaluación
    match (expected_tool, bool(tool_calls), finish):
        # Conversacional: no se esperaba herramienta
        case (None, False, _):
            return True, "Respuesta conversacional correcta (sin herramientas)", None, details

        # Conversacional: modelo alucinó una herramienta
        case (None, True, _) | (None, _, "tool_calls"):
            called_name = (
                tool_calls[0].get("function", {}).get("name") if tool_calls else "desconocida"
            )
            hint = (
                f"PISTA: El modelo alucinó una llamada a '{called_name}' "
                f"para una pregunta conversacional."
            )
            return False, f"Llamó a herramienta '{called_name}' inesperadamente", hint, details

        # Se esperaba tool_call pero no llegó
        case (_, False, _) if finish != "tool_calls":
            preview = (content[:150].replace("\n", " ") if content else "vacío")
            hint = (
                f"PISTA: El modelo respondió con texto plano (finish={finish}).\n"
                f"   Texto: '{preview}'...\n"
                f"   Posibles causas: plantilla Jinja sin soporte tool-calling, "
                f" temperatura alta, o el modelo no entendió la necesidad de usar una herramienta."
            )
            return False, f"finish_reason={finish} (esperado 'tool_calls')", hint, details

        # tool_calls vacío
        case (_, False, "tool_calls"):
            hint = "PISTA: finish_reason='tool_calls' pero el array está vacío."
            return False, "Estructura de tool_calls vacía", hint, details

        # Caso normal: evaluar tool_call
        case _:
            return _evaluate_single_tool(case, tool_calls, details)


def _evaluate_single_tool(
    case: TestCase, tool_calls: list[dict[str, Any]], details: dict[str, Any]
) -> EvalResult:
    """Evalúa un único tool_call esperado."""
    name, args, parse_err = _parse_tool_call(tool_calls[0])
    details["tool_name"] = name
    details["parsed_args"] = args
    expected_tool: str = case["expected_tool"]

    if parse_err:
        hint = (
            f"PISTA: El modelo intentó llamar a '{name}' pero generó "
            f"argumentos JSON inválidos.\n   {parse_err}"
        )
        return False, f"JSON de argumentos malformado en '{name}'", hint, details

    if name != expected_tool:
        hint = (
            f"PISTA: El modelo seleccionó '{name}'. Se esperaba '{expected_tool}'.\n"
            f"   Revisa las descripciones en TOOLS para reducir ambigüedad."
        )
        return False, f"Herramienta incorrecta: '{name}' (esperada '{expected_tool}')", hint, details

    hint = _check_args(name, args, case["expected_args_subset"])
    if hint:
        return False, f"Parámetro incorrecto en '{name}'", hint, details

    return True, f"Correcto → {name}({json.dumps(args, ensure_ascii=False)})", None, details


def _evaluate_sequential(
    case: TestCase,
    tool_calls: list[dict[str, Any]],
    finish: str | None,
    details: dict[str, Any],
) -> EvalResult:
    """Evalúa una secuencia de tool_calls esperada."""
    expected_seq: list[dict[str, Any]] = case["expected_tool_sequence"]

    if not tool_calls:
        hint = (
            f"PISTA: Se esperaban {len(expected_seq)} tool_calls pero no se recibió ninguna.\n"
            f"   finish_reason={finish}"
        )
        return False, "No se generaron tool_calls en secuencia", hint, details

    if len(tool_calls) < len(expected_seq):
        hint = (
            f"PISTA: Se esperaban {len(expected_seq)} tool_calls "
            f"pero solo se recibieron {len(tool_calls)}."
        )
        return False, f"Secuencia incompleta ({len(tool_calls)}/{len(expected_seq)})", hint, details

    # Evaluar cada paso con zip (Python 3.14 optimiza zip con iterables cortos)
    all_calls_info: list[dict[str, Any]] = []

    for i, (actual_call, expected) in enumerate(zip(tool_calls, expected_seq, strict=True)):
        name, args, parse_err = _parse_tool_call(actual_call)
        all_calls_info.append({"tool": name, "args": args})

        if parse_err:
            return False, f"Paso {i + 1}: JSON inválido", f"PISTA: Paso {i+1} - JSON inválido para '{name}'.", details

        if name != expected["tool"]:
            hint = f"PISTA: Paso {i + 1} - Se esperaba '{expected['tool']}' pero se obtuvo '{name}'."
            return False, f"Paso {i + 1}: herramienta incorrecta '{name}'", hint, details

        hint = _check_args(name, args, expected["args_subset"], step=i + 1)
        if hint:
            return False, f"Paso {i + 1}: argumento incorrecto en '{name}'", hint, details

    summary = " → ".join(f"{c['tool']}(...)" for c in all_calls_info)
    return True, f"Secuencia correcta: {summary}", None, details


# ─── Exportación (pathlib) ───────────────────────────────────────────────────

def export_results(
    results: list[dict[str, Any]],
    cfg: Config,
    passed: int,
    failed: int,
    total_time: float,
) -> None:
    """Guarda los resultados en un archivo JSON usando pathlib."""
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "server": cfg.server,
        "model": cfg.model,
        "repeat": cfg.repeat,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed / max(len(results), 1)) * 100:.1f}%",
            "total_time_s": round(total_time, 2),
        },
        "results": results,
    }

    path = Path(cfg.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cprint(Color.GREEN, f"  Reporte exportado → {path}")


# ─── Estadísticas de categoría ────────────────────────────────────────────────

def _update_category_stats(
    stats: dict[str, dict[str, int]], category: str, passed: bool
) -> None:
    """Actualiza stats de categoría de forma funcional."""
    stats.setdefault(category, {"passed": 0, "total": 0})
    stats[category]["total"] += 1
    if passed:
        stats[category]["passed"] += 1


def _category_summary_line(cat: str, stats: dict[str, int]) -> str:
    """Genera una línea de resumen por categoría."""
    total = stats["total"]
    passed = stats["passed"]
    pct = (passed / max(total, 1)) * 100
    icon = "✓" if pct == 100 else "✗" if pct == 0 else "~"
    return f"    {icon} {cat:<15}: {passed}/{total} ({pct:.0f}%)"


# ─── Ejecución de pruebas ────────────────────────────────────────────────────

def run_all(cfg: Config) -> int:
    # Filtrar test cases por ID usando frozenset
    cases: list[TestCase] = (
        [c for c in TEST_CASES if c["id"] in cfg.tests]
        if cfg.tests
        else TEST_CASES
    )

    if not cases:
        cprint(Color.RED, f"  Ningún test encontrado para IDs: {cfg.tests}")
        return 1

    total_cases = len(cases) * cfg.repeat

    print("=" * 80)
    cprint(Color.BOLD, "  TOOL CALLING BENCHMARK — EDICIÓN EXTENDIDA & DIAGNÓSTICA")
    print(f"  Servidor:    {cfg.server}")
    print(f"  Modelo:      {cfg.model}")
    print(f"  Herramientas: {len(TOOLS)}")
    rep_info = f" (×{cfg.repeat} = {total_cases} total)" if cfg.repeat > 1 else ""
    print(f"  Pruebas:     {len(cases)}{rep_info}")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    total_time = 0.0
    all_results: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, int]] = {}

    for iteration in range(1, cfg.repeat + 1):
        if cfg.repeat > 1:
            cprint(Color.CYAN, f"  ─── Iteración {iteration}/{cfg.repeat} ───")
            print()

        for case in cases:
            cid: int = case["id"]
            desc: str = case["desc"]
            cat: str = case.get("category", "General")
            is_seq: bool = case.get("sequential", False)
            prompt_user: str = case["messages"][-1]["content"]

            _update_category_stats(category_stats, cat, False)  # temp increment total

            # Encabezado del test
            print("─" * 80)
            test_label = f"Prueba {cid}/{len(cases)}"
            if cfg.repeat > 1:
                test_label += f" (iter {iteration})"
            cprint(Color.BOLD, f"  [{test_label}] [{cat}] {desc}")
            print(f'  Prompt: "{prompt_user}"')

            if is_seq:
                seq_names = [s["tool"] for s in case["expected_tool_sequence"]]
                print(f"  Secuencia esperada: {' → '.join(seq_names)}")
            else:
                exp = case.get("expected_tool")
                print(f"  Herramienta esperada: {exp or 'Ninguna (Conversacional)'}")

            # Llamada al servidor
            resp, elapsed, err_msg = chat_completion(cfg, case["messages"], TOOLS)
            total_time += elapsed

            # Evaluación
            ok, msg, hint, details = evaluate(case, resp, err_msg)

            # Output en tiempo real
            print(f"  Tiempo: {elapsed:.2f}s")
            if details["finish_reason"]:
                print(f"  Finish reason: {details['finish_reason']}")

            match details:
                case {"tool_name": name, "parsed_args": args} if name:
                    print(f"  Herramienta: {name}")
                    print(f"  Argumentos: {json.dumps(args, ensure_ascii=False)}")
                case {"raw_content": content} if content:
                    short = content.replace("\n", " ")[:120]
                    if len(content) > 120:
                        short += "..."
                    print(f'  Texto: "{short}"')

            # Resultado
            if ok:
                passed += 1
                category_stats[cat]["passed"] += 1
                cprint(Color.GREEN, f"  RESULTADO: APROBADO → {msg}")
            else:
                failed += 1
                cprint(Color.RED, f"  RESULTADO: FALLIDO → {msg}")
                if hint:
                    cprint(Color.YELLOW, f"  {hint}")

            all_results.append({
                "id": cid,
                "desc": desc,
                "category": cat,
                "iteration": iteration if cfg.repeat > 1 else None,
                "passed": ok,
                "message": msg,
                "elapsed_s": round(elapsed, 3),
                "finish_reason": details["finish_reason"],
                "tool_name": details["tool_name"],
                "parsed_args": details["parsed_args"],
                "raw_content": details["raw_content"],
                "hint": hint,
            })

            print()

    # ─── Resumen Final ───
    print("=" * 80)
    cprint(Color.BOLD, "  RESUMEN FINAL")
    print("=" * 80)

    total = passed + failed
    rate = (passed / max(total, 1)) * 100
    avg_time = total_time / max(total, 1)

    cprint(
        Color.GREEN if failed == 0 else Color.YELLOW,
        f"  Aprobadas: {passed}/{total} ({rate:.1f}%)",
    )
    if failed:
        cprint(Color.RED, f"  Fallidas:  {failed}/{total}")
    print(f"  Tiempo total HTTP: {total_time:.2f}s (promedio: {avg_time:.2f}s/prueba)")

    if category_stats:
        print()
        print("  Desglose por categoría:")
        lines = [_category_summary_line(cat, stats) for cat, stats in category_stats.items()]
        print("\n".join(lines))

    print("=" * 80)

    if cfg.output:
        export_results(all_results, cfg, passed, failed, total_time)

    return 0 if failed == 0 else 1


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Tool Calling Benchmark para llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080/v1",
        help="URL base del servidor (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        default="cualquiera",
        help="Nombre del modelo a reportar (default: cualquiera)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout por petición en segundos (default: 60)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="IDs de tests a ejecutar, separados por coma (ej: 1,3,5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta para exportar resultados en JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar información de debug adicional",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Número de veces a repetir cada test (default: 1)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Máximo de reintentos por petición (default: 3)",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar/desactivar colores ANSI (default: auto-detect)",
    )

    args = parser.parse_args()

    # Auto-detect colores
    if args.color and not sys.stdout.isatty():
        args.color = False

    # Parsear tests IDs a frozenset
    test_ids: frozenset[int] | None = None
    if args.tests:
        try:
            test_ids = frozenset(int(x.strip()) for x in args.tests.split(","))
        except ValueError:
            parser.error(f"--tests debe ser enteros separados por coma: '{args.tests}'")

    cfg = Config(
        server=args.server,
        model=args.model,
        timeout=args.timeout,
        verbose=args.verbose,
        color=args.color,
        output=args.output,
        tests=test_ids,
        repeat=args.repeat,
        max_retries=args.retries,
    )

    if not cfg.color:
        Color.disable()

    return cfg


def main() -> None:
    cfg = parse_args()
    sys.exit(run_all(cfg))


if __name__ == "__main__":
    main()
