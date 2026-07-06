#!/usr/bin/env python3
"""
Definiciones compartidas de herramientas (tool schemas) para los tests de tool calling.
Fuente única de verdad — todos los test-*.py importan desde aquí.
"""

from typing import Any

# ─── Type aliases (PEP 695 — Python 3.12+) ──────────────────────────────────

type ToolDef = dict[str, Any]
type ArgsSubset = dict[str, str | int | float | bool]


# ─── Catálogo de herramientas (14) ───────────────────────────────────────────

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
                actual_str = str(actual).strip()
                expected_str = expected_val.strip()
                strict_keys = {"path", "to", "subject", "url", "ticker", "command"}
                strict_tools = {"system_command", "write_file", "read_file", "fetch_web_page", "send_email"}
                if key in strict_keys or tool_name in strict_tools:
                    if actual_str != expected_str:
                        return (
                            f"{step_prefix}PISTA: '{key}' = '{actual}', "
                            f"se esperaba exactamente '{expected_val}'."
                        )
                elif expected_str.lower() not in actual_str.lower():
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
