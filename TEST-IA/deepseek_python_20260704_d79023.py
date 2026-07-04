#!/usr/bin/env python3
"""
Evaluación de modelos de IA con llama.cpp.
Ejecuta una suite de 19 pruebas y genera un informe detallado.

Python 3.14+ requerido.
"""

import argparse
import ast
import json
import os
import re
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Never, Self
from collections.abc import Callable

import requests

# ─── Type aliases (PEP 695) ─────────────────────────────────────────────────

type ValidationTuple = tuple[bool, str, float]
type TestResult = dict[str, Any]
type ResultsMap = dict[str, TestResult]
type MetricsMap = dict[str, int | str]
type ValidatorFn = Callable[[str], ValidationTuple]


# ─── Configuración (frozen dataclass) ───────────────────────────────────────

@dataclass(slots=True, frozen=True)
class ServerConfig:
    host: str
    chat_endpoint: str
    metrics_endpoint: str
    timeout: int
    max_tokens: int
    tests_filter: str | None = None
    quiet: bool = False
    log_file: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        host = os.getenv("LLAMA_HOST", "http://127.0.0.1:8080")
        return cls(
            host=host,
            chat_endpoint=f"{host}/v1/chat/completions",
            metrics_endpoint=f"{host}/metrics",
            timeout=int(os.getenv("TIMEOUT", "120")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            tests_filter=os.getenv("TESTS_FILTER"),
            quiet=os.getenv("QUIET", "0").lower() in ("1", "true", "yes"),
            log_file=os.getenv("LOG_FILE"),
        )


# ─── Colores ANSI ────────────────────────────────────────────────────────────

class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


NO_COLOR = os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb"


def c(color: str, text: str) -> str:
    if NO_COLOR:
        return text
    return f"{color}{text}{Color.RESET}"


# ─── Funciones auxiliares ────────────────────────────────────────────────────

def sanitizar_nombre(nombre: str) -> str:
    """Convierte un nombre de modelo en un string seguro para filenames."""
    nombre = nombre.strip()
    nombre = re.sub(r'[^\w\s\-]', '', nombre)
    nombre = re.sub(r'\s+', '_', nombre)
    nombre = nombre.strip('_')
    return nombre[:80] if nombre else "modelo_desconocido"


def extraer_codigo_python(respuesta: str) -> str:
    """Extrae el contenido de un bloque de código python de Markdown."""
    patron = r'```python\s*\n(.*?)\n```'
    match = re.search(patron, respuesta, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    patron2 = r'```\s*\n(.*?)\n```'
    match2 = re.search(patron2, respuesta, re.DOTALL)
    if match2:
        return match2.group(1)
    return respuesta


def extraer_codigo_bash(respuesta: str) -> str:
    """Extrae el contenido de un bloque de código bash de Markdown."""
    patron = r'```(?:bash|sh)?\s*\n(.*?)\n```'
    match = re.search(patron, respuesta, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return respuesta


def extraer_json(respuesta: str) -> str:
    """
    Extrae la cadena JSON limpia de una respuesta que puede contener Markdown.
    Busca bloque ```json, luego intenta respuesta completa, y como último
    recurso busca el primer objeto/array JSON válido.
    """
    # 1. Buscar bloque de código ```json ... ```
    patron_bloque = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(patron_bloque, respuesta, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Intentar parsear la respuesta completa como JSON
    try:
        json.loads(respuesta.strip())
        return respuesta.strip()
    except json.JSONDecodeError:
        pass

    # 3. Buscar objeto/array JSON con balanceado de llaves/corchetes
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = respuesta.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(respuesta)):
            ch = respuesta[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = respuesta[start_idx:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    return respuesta.strip()


def calcular_score(parte_a: bool, parte_b: bool) -> float:
    """Calcula un score numérico: 1.0 ambas partes, 0.5 una parte, 0.0 ninguna."""
    match (parte_a, parte_b):
        case (True, True):
            return 1.0
        case (True, False) | (False, True):
            return 0.5
        case _:
            return 0.0


# ─── Funciones de validación ─────────────────────────────────────────────────

def validar_bash(respuesta: str) -> ValidationTuple:
    codigo = extraer_codigo_bash(respuesta)
    tiene_rsync = "rsync" in codigo
    tiene_log = any(p in codigo.lower() for p in ("log", "logger", "logrotate"))
    tiene_error = any(p in codigo.lower() for p in ("error", "retry", "trap", "exit"))
    tiene_shebang = bool(re.search(r'^\s*#!', codigo, re.MULTILINE))

    partes = [tiene_rsync, tiene_log, tiene_error, tiene_shebang]
    score = sum(partes) / len(partes)

    faltan = []
    if not tiene_rsync:
        faltan.append("rsync")
    if not tiene_log:
        faltan.append("rotación de logs")
    if not tiene_error:
        faltan.append("manejo de errores")
    if not tiene_shebang:
        faltan.append("shebang #!/bin/bash")

    msg = ", ".join(faltan) if faltan else "Script bash completo: rsync, logs, errores, shebang."
    return score == 1.0, msg, round(score, 2)


def validar_python(respuesta: str) -> ValidationTuple:
    codigo = extraer_codigo_python(respuesta)
    if not codigo.strip():
        return False, "No se encontró código Python.", 0.0
    try:
        ast.parse(codigo)
        tiene_imports = "import " in codigo or "from " in codigo
        score = 1.0 if tiene_imports else 0.8
        msg = "Código Python sintácticamente válido."
        if not tiene_imports:
            msg += " (sin imports detectados)"
        return True, msg, score
    except SyntaxError as e:
        return False, f"Error de sintaxis: {e}", 0.0


def validar_json(respuesta: str) -> ValidationTuple:
    try:
        json_limpio = extraer_json(respuesta)
        data = json.loads(json_limpio)
        if not isinstance(data, dict):
            return False, "JSON no es un objeto.", 0.3

        # Verificar estructura de tool calling
        nombre_funcion = None
        argumentos = None

        # Formato OpenAI: {"function": {"name": ..., "arguments": {...}}}
        if "function" in data and isinstance(data["function"], dict):
            nombre_funcion = data["function"].get("name", "")
            argumentos = data["function"].get("arguments", {})
        # Formato directo: {"name": ..., "arguments": {...}}
        elif "name" in data:
            nombre_funcion = data.get("name", "")
            argumentos = data.get("arguments", {})
        # Formato tool: {"tool": ..., "input": {...}}
        elif "tool" in data:
            nombre_funcion = data.get("tool", "")
            argumentos = data.get("input", data.get("arguments", {}))

        if not nombre_funcion:
            return True, "JSON válido pero sin nombre de función.", 0.3

        # Verificar que la función sea la esperada
        funcion_esperada = "buscar_texto_en_archivos"
        if nombre_funcion.lower() != funcion_esperada.lower():
            return True, f"JSON válido pero función inesperada: '{nombre_funcion}'.", 0.5

        # Verificar argumentos requeridos
        if not isinstance(argumentos, dict):
            return True, "JSON válido pero argumentos no son objeto.", 0.5

        args_lower = {k.lower(): v for k, v in argumentos.items()}
        campos_ok = sum(1 for c in ("directorio", "extension", "texto_busqueda") if c in args_lower)
        score = 0.5 + (campos_ok / 3) * 0.5

        if campos_ok == 3:
            return True, "Tool call completa: función y 3 argumentos.", 1.0
        elif campos_ok > 0:
            return True, f"Tool call parcial: {campos_ok}/3 argumentos.", score
        else:
            return True, "Tool call sin argumentos esperados.", 0.5

    except json.JSONDecodeError:
        return False, "No es un JSON válido.", 0.0


def validar_sql(respuesta: str) -> ValidationTuple:
    resp_upper = respuesta.upper()

    # Detectar window functions
    tiene_over = bool(re.search(r'\bOVER\s*\(', respuesta, re.IGNORECASE))
    tiene_row_number = bool(re.search(r'\bROW_NUMBER\s*\(', respuesta, re.IGNORECASE))
    tiene_rank = bool(re.search(r'\b(?:RANK|DENSE_RANK)\s*\(', respuesta, re.IGNORECASE))
    tiene_partition = bool(re.search(r'\bPARTITION\s+BY\b', respuesta, re.IGNORECASE))
    window_func = tiene_over and (tiene_row_number or tiene_rank)

    # Detectar estructura SQL válida
    tiene_select = bool(re.search(r'\bSELECT\b', respuesta, re.IGNORECASE))
    tiene_from = bool(re.search(r'\bFROM\b', respuesta, re.IGNORECASE))
    tiene_join = bool(re.search(r'\b(?:INNER\s+)?JOIN\b', respuesta, re.IGNORECASE))
    tiene_group = bool(re.search(r'\bGROUP\s+BY\b', respuesta, re.IGNORECASE))

    # Verificar referencias a tablas del test
    tiene_empleados = bool(re.search(r'\bempleados?\b', respuesta, re.IGNORECASE))
    tiene_departamentos = bool(re.search(r'\bdepartamentos?\b', respuesta, re.IGNORECASE))

    # Verificar columna salario
    tiene_salario = bool(re.search(r'\bsalario\b', respuesta, re.IGNORECASE))

    # Calcular score
    componentes = [window_func, tiene_select, tiene_from, tiene_empleados or tiene_departamentos]
    score = sum(componentes) / len(componentes)

    # Bonus por PARTITION BY y tablas correctas
    if window_func and tiene_partition and tiene_salario:
        score = min(score + 0.2, 1.0)

    if window_func and tiene_from:
        msg = "Window functions con OVER y FROM detectados."
        if tiene_partition:
            msg += " PARTITION BY presente."
        return True, msg, score
    elif tiene_over and tiene_from:
        return True, "OVER detectado pero sin función de ventana clara.", score * 0.7
    elif tiene_from:
        return False, "FROM detectado pero sin window functions.", 0.3
    else:
        return False, "No se detectaron window functions ni FROM.", 0.0


def validar_pytest(respuesta: str) -> ValidationTuple:
    tiene_assert = "assert" in respuesta
    tiene_mark = "@pytest.mark" in respuesta or "pytest.mark" in respuesta
    tiene_func_test = bool(re.search(r"def test_", respuesta))

    partes = [tiene_assert, tiene_mark, tiene_func_test]
    score = sum(partes) / len(partes)

    faltan = [
        nombre for tiene, nombre in [
            (tiene_assert, "aserciones"),
            (tiene_mark, "marcadores pytest"),
            (tiene_func_test, "convención def test_"),
        ]
        if not tiene
    ]

    if faltan:
        return False, f"Faltan: {', '.join(faltan)}", score
    return True, "Completo: aserciones, marcadores, convención test_.", score


def validar_go(respuesta: str) -> ValidationTuple:
    tiene_package = "package main" in respuesta
    tiene_func = "func main" in respuesta
    tiene_import = "import" in respuesta

    partes = [tiene_package, tiene_func, tiene_import]
    score = sum(partes) / len(partes)

    if tiene_package and tiene_func:
        msg = "Estructura Go válida (package + func main)."
        if not tiene_import:
            msg += " (sin imports detectados)"
        return True, msg, score

    faltan = [
        nombre for tiene, nombre in [
            (tiene_package, "package main"),
            (tiene_func, "func main"),
        ]
        if not tiene
    ]
    return False, f"Faltan: {', '.join(faltan)}.", score


def validar_rust(respuesta: str) -> ValidationTuple:
    codigo_match = re.search(r'```(?:rust)?\s*\n(.*?)\n```', respuesta, re.DOTALL | re.IGNORECASE)
    codigo = codigo_match.group(1) if codigo_match else respuesta

    tiene_fn = bool(re.search(r'\bfn\s+\w+', codigo))
    tiene_result = bool(re.search(r'Result\s*<', codigo))
    tiene_option = bool(re.search(r'Option\s*<', codigo))
    tiene_use = bool(re.search(r'\buse\s+', codigo))
    tiene_return_arrow = '->' in codigo

    score = 0.0
    if tiene_fn:
        score += 0.3
    if tiene_result or tiene_option:
        score += 0.3
    if tiene_return_arrow:
        score += 0.2
    if tiene_use:
        score += 0.2
    if tiene_fn and tiene_result and (tiene_use or tiene_return_arrow):
        score = 1.0

    partes = []
    if tiene_fn:
        partes.append("Funciones fn detectadas")
    else:
        return False, "No se encontraron funciones 'fn'.", 0.0

    if tiene_result or tiene_option:
        partes.append("Result/Option")
    else:
        partes.append("Sin Result/Option")

    if tiene_use:
        partes.append(", use detectado")

    msg = ". ".join(partes) + f". Score: {score:.2f}"
    return True, msg, score


def validar_js(respuesta: str) -> ValidationTuple:
    tiene_http = "fetch" in respuesta or "axios" in respuesta
    tiene_async = "async" in respuesta or "await" in respuesta or ".then" in respuesta
    tiene_funcion = "function " in respuesta or "const " in respuesta or "=>" in respuesta

    partes = [tiene_http, tiene_async, tiene_funcion]
    score = sum(partes) / len(partes)

    match (tiene_http, tiene_async):
        case (True, True):
            return True, "HTTP + async/await detectados.", score
        case (True, False):
            return True, "HTTP sin async detectado.", 0.5
        case _:
            return False, "No se detectaron peticiones HTTP.", 0.0


def validar_traduccion(respuesta: str) -> ValidationTuple:
    patron_es = r'[áéíóúñÑ¿¡]'
    tiene_tilde = bool(re.search(patron_es, respuesta))
    tiene_longitud = len(respuesta.strip()) > 20

    score = calcular_score(tiene_tilde, tiene_longitud)

    match (tiene_tilde, tiene_longitud):
        case (True, True):
            return True, "Texto en español con longitud adecuada.", score
        case (True, False):
            return True, "Español detectado pero respuesta muy corta.", 0.5
        case (False, True):
            return False, "Longitud adecuada pero sin caracteres españoles.", 0.5
        case _:
            return False, "No se detectaron caracteres españoles ni longitud suficiente.", 0.0


def validar_seguridad(respuesta: str) -> ValidationTuple:
    resp_lower = respuesta.lower()

    # Detectar identificación de vulnerabilidad
    vuln_identificada = any(v in resp_lower for v in (
        "sql injection", "inyección sql", "inyeccion sql", "sqli",
        "vulnerable", "vulnerabilidad"
    ))

    # Detectar código seguro real (no solo mención)
    tiene_prepared = bool(re.search(r'(prepare|prepared|stmt|statement)', resp_lower))
    tiene_parameterized = bool(re.search(r'(parametriz|parameterized|\?\s*,|%\s*s)', resp_lower))
    tiene_bind = bool(re.search(r'(bind_param|bindvalue|bind_value|\$\w+\s*=)', resp_lower))
    tiene_escape = bool(re.search(r'(escape|sanitiz|mysql_real_escape|htmlspecialchars)', resp_lower))

    # Verificar que haya CÓDIGO seguro, no solo teoría
    tiene_codigo_php = bool(re.search(r'(\<\?php|\$conn|\$stmt|mysqli|pdo)', respuesta, re.IGNORECASE))
    tiene_ejemplo = tiene_prepared or tiene_parameterized or tiene_bind or tiene_escape

    score = calcular_score(vuln_identificada, tiene_ejemplo and tiene_codigo_php)

    if vuln_identificada and tiene_ejemplo and tiene_codigo_php:
        return True, "Vulnerabilidad identificada y mitigación con código seguro.", 1.0
    elif vuln_identificada and tiene_ejemplo:
        return True, "Vulnerabilidad y mitigación detectadas (sin código PHP).", 0.7
    elif vuln_identificada:
        return True, "Vulnerabilidad identificada pero mitigación débil.", 0.3
    elif tiene_ejemplo:
        return True, "Mitigación sin identificar vulnerabilidad explícita.", 0.5
    else:
        return False, "No se detectaron vulnerabilidades ni mitigaciones.", 0.0


def validar_mutabilidad(respuesta: str) -> ValidationTuple:
    if "def agregar_item" not in respuesta:
        return False, "No se encontró la función 'agregar_item'.", 0.0

    usa_none = "lista=None" in respuesta or "lista = None" in respuesta
    tiene_check_none = (
        "if lista is None" in respuesta
        or "if lista==None" in respuesta
        or "lista = []" in respuesta
    )

    match (usa_none, tiene_check_none):
        case (True, True):
            return True, "Mutabilidad corregida: lista=None + inicialización condicional.", 1.0
        case (True, False):
            return True, "Usa lista=None pero falta inicialización explícita.", 0.7
        case _ if "lista=[]" in respuesta:
            return False, "Todavía usa lista=[] como valor por defecto (mutabilidad).", 0.0
        case _:
            return False, "No se detecta corrección del bug de mutabilidad.", 0.0


def validar_optimizacion(respuesta: str) -> ValidationTuple:
    codigo = extraer_codigo_python(respuesta)

    # Verificar uso real de set/dict (no solo en comentarios)
    lineas_codigo = [l for l in codigo.split('\n') if l.strip() and not l.strip().startswith('#')]
    codigo_sin_comentarios = '\n'.join(lineas_codigo)

    usa_set = bool(re.search(r'\bset\s*\(', codigo_sin_comentarios))
    usa_dict = bool(re.search(r'\bdict\s*\(|\{[^}]*for\s+\w+\s+in', codigo_sin_comentarios))
    usa_comprension = bool(re.search(r'\{[^}]*for\s+\w+\s+in[^}]*\}', codigo_sin_comentarios))
    usa_counter = 'Counter' in codigo_sin_comentarios

    # Verificar que no haya bucles anidados O(N^2)
    tiene_bucles_anidados = bool(re.search(
        r'for\s+\w+.*:.*\n\s+for\s+\w+', codigo_sin_comentarios
    ))

    optimizado = usa_set or usa_dict or usa_comprension or usa_counter

    if optimizado:
        metodo = []
        if usa_set:
            metodo.append("set()")
        if usa_dict or usa_comprension:
            metodo.append("dict/comprensión")
        if usa_counter:
            metodo.append("Counter")
        msg = f"Optimización O(N) detectada: {', '.join(metodo)}."
        if tiene_bucles_anidados:
            msg += " (⚠️ posible bucle anidado residual)"
            return True, msg, 0.7
        return True, msg, 1.0
    else:
        return False, "No se detecta uso de set/dict/Counter; podría seguir siendo O(N²).", 0.0


def validar_retry(respuesta: str) -> ValidationTuple:
    codigo = extraer_codigo_python(respuesta)

    # Verificar implementación real de retry (no solo comentarios)
    lineas_codigo = [l for l in codigo.split('\n') if l.strip() and not l.strip().startswith('#')]
    codigo_real = '\n'.join(lineas_codigo)

    # Buscar loop/bucle de reintentos
    tiene_loop = bool(re.search(r'for\s+\w+\s+in\s+range\s*\(', codigo_real))
    tiene_while = bool(re.search(r'while\s+', codigo_real))

    # Variable de control de intentos
    tiene_retry_var = bool(re.search(r'(retry|attempt|intento|intentos?)\s*[=<>]', codigo_real, re.IGNORECASE))

    # Backoff exponencial
    tiene_sleep = 'sleep' in codigo_real.lower() or 'time.sleep' in codigo_real
    tiene_backoff = bool(re.search(r'(backoff|exponencial|2\s*\*\*|2\s*\^\s*\w+|\*\s*2)', codigo_real, re.IGNORECASE))

    # Verificar que haya un mecanismo de reintento real
    tiene_mecanismo = tiene_loop or tiene_while or tiene_retry_var

    if tiene_mecanismo and tiene_sleep and tiene_backoff:
        return True, "Reintentos con backoff exponencial implementado.", 1.0
    elif tiene_mecanismo and tiene_sleep:
        return True, "Reintentos con sleep detectados (sin backoff claro).", 0.7
    elif tiene_mecanismo:
        return True, "Mecanismo de reintento detectado (sin sleep/backoff).", 0.5
    elif tiene_retry_var or tiene_sleep:
        return True, "Componentes de retry parciales.", 0.3
    else:
        return False, "No se detecta implementación real de reintentos.", 0.0


def validar_extraccion_info(respuesta: str) -> ValidationTuple:
    try:
        json_limpio = extraer_json(respuesta)
        data = json.loads(json_limpio)
        if not isinstance(data, dict):
            return False, "JSON no es un diccionario.", 0.0

        # Claves esperadas (singular y plural)
        grupos = {
            "personas": ["personas", "persona", "people", "names", "nombre", "nombres"],
            "lugares": ["lugares", "lugar", "places", "location", "ubicacion"],
            "fechas": ["fechas", "fecha", "dates", "date"],
            "organizaciones": ["organizaciones", "organizacion", "organizations", "company", "empresas"],
        }

        claves_encontradas = {}
        for grupo, variantes in grupos.items():
            for variante in variantes:
                if variante in data:
                    claves_encontradas[grupo] = data[variante]
                    break

        if not claves_encontradas:
            return False, "JSON sin entidades esperadas.", 0.0

        # Verificar que los valores sean listas y no vacías
        valores_ok = 0
        for grupo, valor in claves_encontradas.items():
            if isinstance(valor, list) and len(valor) > 0:
                valores_ok += 1
            elif isinstance(valor, str) and valor.strip():
                valores_ok += 1

        total_grupos = len(grupos)
        score = (len(claves_encontradas) / total_grupos) * 0.5 + (valores_ok / len(claves_encontradas)) * 0.5

        # Verificar contenido específico del test
        contenido_texto = json.dumps(data).lower()
        tiene_openai = "openai" in contenido_texto
        tiene_san_francisco = "san francisco" in contenido_texto
        tiene_sam_altman = "sam altman" in contenido_texto or "altman" in contenido_texto
        tiene_2023 = "2023" in contenido_texto

        entidades_especificas = sum([tiene_openai, tiene_san_francisco, tiene_sam_altman, tiene_2023])

        if valores_ok >= 3 and entidades_especificas >= 2:
            return True, f"Entidades completas: {', '.join(claves_encontradas.keys())}.", 1.0
        elif valores_ok >= 2:
            return True, f"Entidades parciales: {', '.join(claves_encontradas.keys())}.", score
        else:
            return True, "Entidades encontradas pero con valores vacíos.", 0.3

    except json.JSONDecodeError:
        return False, "No es un JSON válido.", 0.0


def validar_tool_calling_avanzado(respuesta: str) -> ValidationTuple:
    try:
        json_limpio = extraer_json(respuesta)
        data = json.loads(json_limpio)
        if not isinstance(data, list):
            return False, "JSON no es una lista.", 0.0
        if not data:
            return False, "Lista vacía.", 0.0

        # Funciones esperadas y sus argumentos clave
        funciones_esperadas = {
            "obtener_clima": ["ciudad", "city", "ubicacion", "location"],
            "calcular_raiz": ["numero", "number", "num", "valor", "value"],
            "enviar_correo": ["email", "correo", "destinatario", "to"],
        }
        funciones_encontradas = set()
        argumentos_correctos = 0
        items_validos = 0

        for item in data:
            if not isinstance(item, dict):
                continue

            # Extraer nombre de función (múltiples formatos)
            nombre = None
            argumentos = None
            if "function" in item and isinstance(item["function"], dict):
                nombre = item["function"].get("name", "")
                argumentos = item["function"].get("arguments", {})
            elif "name" in item:
                nombre = item.get("name", "")
                argumentos = item.get("arguments", item.get("input", {}))
            elif "tool" in item:
                nombre = item.get("tool", "")
                argumentos = item.get("input", item.get("arguments", {}))

            if nombre:
                items_validos += 1
                nombre_lower = nombre.lower()

                # Buscar coincidencia con funciones esperadas
                for func_esperada, args_keys in funciones_esperadas.items():
                    if func_esperada in nombre_lower or nombre_lower in func_esperada:
                        funciones_encontradas.add(func_esperada)
                        # Verificar argumentos
                        if isinstance(argumentos, dict):
                            args_lower = {k.lower(): v for k, v in argumentos.items()}
                            if any(ak in args_lower for ak in args_keys):
                                argumentos_correctos += 1
                        break

        total_esperadas = len(funciones_esperadas)
        funciones_match = len(funciones_encontradas)
        score = (funciones_match / total_esperadas) * 0.6 + (argumentos_correctos / total_esperadas) * 0.4

        if funciones_match == total_esperadas and argumentos_correctos == total_esperadas:
            return True, f"Lista válida: {total_esperadas} tool calls con argumentos.", 1.0
        elif funciones_match == total_esperadas:
            return True, f"Funciones correctas pero faltan argumentos ({argumentos_correctos}/{total_esperadas}).", score
        elif funciones_match > 0:
            faltan = funciones_esperadas.keys() - funciones_encontradas
            return True, f"Parcial: {funciones_match}/{total_esperadas} funciones. Faltan: {', '.join(faltan)}.", score
        else:
            return False, f"Ninguna función esperada encontrada ({items_validos} items).", 0.0

    except json.JSONDecodeError:
        return False, "No es un JSON válido.", 0.0


def validar_logica(respuesta: str) -> ValidationTuple:
    resp_lower = respuesta.lower()

    # Detectar personajes
    tiene_pollo = "pollo" in resp_lower
    tiene_zorro = "zorro" in resp_lower
    tiene_grano = any(w in resp_lower for w in ("maíz", "maiz", "grano", "trigo", "grano de maíz"))
    tiene_personajes = tiene_pollo and tiene_zorro and tiene_grano

    # Detectar solución paso a paso (el pollo va primero)
    tiene_paso1 = bool(re.search(
        r'(paso\s*1|primero|1[°º\.]|en\s+el\s+primer|llev[ao]|cruz[ao]).{1,40}pollo',
        resp_lower
    ))

    # Detectar retorno del pollo (paso crítico)
    tiene_retorno = bool(re.search(
        r'(regres|volv|tra|retorn|cruz[ao]\s+de\s+nuevo|segundo\s+viaje|devuelt).{1,40}pollo',
        resp_lower
    ))

    # Detectar que zorro y grano quedan separados
    tiene_separacion = bool(re.search(
        r'(zorro.{1,20}grano|grano.{1,20}zorro|deja.{1,20}zorro.{1,20}grano)',
        resp_lower
    ))

    # Verificar que la secuencia sea lógica
    partes = [tiene_personajes, tiene_paso1, tiene_retorno]
    score = sum(partes) / len(partes)

    if tiene_personajes and tiene_retorno:
        return True, "Razonamiento lógico correcto: personajes y retorno del pollo detectados.", score
    elif tiene_personajes and tiene_paso1:
        return True, "Personajes correctos y primer paso, falta retorno del pollo.", score * 0.8
    elif tiene_personajes:
        return False, "Falta la lógica del retorno del pollo.", score
    else:
        return False, "No se identificaron los personajes o la lógica del acertijo.", score


def validar_matematicas(respuesta: str) -> ValidationTuple:
    resp_lower = respuesta.lower()
    tiene_metodo = any(m in resp_lower for m in (
        "separac", "separable", "integr", "variable separable",
        "separación de variables", "separacion de variables"
    ))

    # Regex flexible para y = C*e^x (múltiples formatos)
    tiene_solucion = bool(re.search(
        r"""(?:y\s*(?:\(x\))?\s*=\s*)?     # opcional: y(x) =
            [CKckABab]\s*                   # constante C/K/A/B
            [*·×]?\s*                       # opcional: operador multiplicación
            (?:e\s*\^?\s*(?:\{?\s*x\s*\}?  # e^x, e^{x}, e^ x
            |exp\s*\(\s*x\s*\))            # exp(x)
            |e\s*\^\s*x)                    # e^x
        """,
        respuesta,
        re.IGNORECASE | re.VERBOSE,
    ))

    # Detección adicional por texto
    if not tiene_solucion:
        tiene_solucion = bool(re.search(
            r'y\s*=\s*[CcKk]\s*\*?\s*e\^?x', respuesta, re.IGNORECASE
        ))

    score = calcular_score(tiene_metodo, tiene_solucion)

    faltan = []
    if not tiene_metodo:
        faltan.append("explicación del método (separación de variables)")
    if not tiene_solucion:
        faltan.append("fórmula de la solución general (y = C*e^x)")

    if score == 1.0:
        return True, "Ecuación resuelta: solución y método correctos.", 1.0
    else:
        return False, f"Faltan: {', '.join(faltan)}.", score


def validar_resumen(respuesta: str) -> ValidationTuple:
    palabras = respuesta.strip().split()
    cant_palabras = len(palabras)
    
    cumple_longitud = 15 <= cant_palabras <= 105
    
    resp_lower = respuesta.lower()
    tiene_tema = "ia" in resp_lower or "inteligencia" in resp_lower
    tiene_detalles = any(w in resp_lower for w in ["ética", "desafío", "responsable", "privacidad", "aplicación", "sociedad", "desarrollo"])
    
    partes_contenido = tiene_tema and tiene_detalles
    
    score = calcular_score(cumple_longitud, partes_contenido)
    
    msg_parts = []
    if cumple_longitud:
        msg_parts.append(f"Longitud adecuada ({cant_palabras} palabras).")
    else:
        msg_parts.append(f"Longitud incorrecta ({cant_palabras} palabras, esperado 15-100).")
        
    if partes_contenido:
        msg_parts.append("Contenido relevante detectado.")
    else:
        msg_parts.append("Falta contenido clave o no parece un resumen del texto.")
        
    return cumple_longitud and partes_contenido, " ".join(msg_parts), score


def validar_explicacion(respuesta: str) -> ValidationTuple:
    resp_lower = respuesta.lower()
    
    conceptos = ["complejidad", "tiempo", "algoritmo", "rendimiento", "peor caso", "crecimiento", "cota", "ejecución"]
    hits_conceptos = sum(1 for c in conceptos if c in resp_lower)
    tiene_explicacion = hits_conceptos >= 2
    
    tiene_o1 = "o(1)" in resp_lower
    tiene_on = "o(n)" in resp_lower or "o( n)" in resp_lower
    tiene_on2 = "o(n^2)" in resp_lower or "o(n²)" in resp_lower or "o(n^{2})" in resp_lower
    tiene_ologn = "o(log n)" in resp_lower or "o(log(n))" in resp_lower or "o(logn)" in resp_lower
    
    ejemplos_encontrados = []
    if tiene_o1: ejemplos_encontrados.append("O(1)")
    if tiene_on: ejemplos_encontrados.append("O(n)")
    if tiene_on2: ejemplos_encontrados.append("O(n^2)")
    if tiene_ologn: ejemplos_encontrados.append("O(log n)")
    
    todos_ejemplos = len(ejemplos_encontrados) == 4
    
    score = calcular_score(tiene_explicacion, todos_ejemplos)
    
    if tiene_explicacion and todos_ejemplos:
        return True, "Explicación completa con todos los ejemplos requeridos (O(1), O(n), O(n^2), O(log n)).", score
    elif todos_ejemplos:
        return False, "Ejemplos correctos, pero explicación conceptual débil.", score
    else:
        faltan = set(["O(1)", "O(n)", "O(n^2)", "O(log n)"]) - set(ejemplos_encontrados)
        msg = f"Explicación incompleta. Faltan ejemplos: {', '.join(faltan)}."
        if not tiene_explicacion:
            msg += " Conceptos clave no detectados."
        return False, msg, score


# ─── Diccionario de validadores ──────────────────────────────────────────────

VALIDATORS: dict[str, ValidatorFn | None] = {
    "1_generacion_codigo": validar_bash,
    "2_algoritmia_compleja": validar_python,
    "3_resolucion_problemas": validar_mutabilidad,
    "4_uso_herramientas_agente": validar_json,
    "5_refactorizacion_y_opt": validar_optimizacion,
    "6_generacion_unit_tests": validar_pytest,
    "7_sql_avanzado": validar_sql,
    "8_api_resiliente_y_errores": validar_retry,
    "9_logica_razonamiento": validar_logica,
    "10_matematicas_ecuacion": validar_matematicas,
    "11_extraccion_info": validar_extraccion_info,
    "12_generacion_go": validar_go,
    "13_generacion_rust": validar_rust,
    "14_generacion_js": validar_js,
    "15_resumen_texto": validar_resumen,
    "16_traduccion": validar_traduccion,
    "17_explicacion_concepto": validar_explicacion,
    "18_tool_calling_avanzado": validar_tool_calling_avanzado,
    "19_seguridad_injection": validar_seguridad,
}


# ─── Suite de pruebas ────────────────────────────────────────────────────────

TEST_SUITE: dict[str, dict[str, str]] = {
    "1_generacion_codigo": {
        "categoria": "Generación de Código",
        "system": "Eres un desarrollador Senior. Responde solo con código limpio y documentado.",
        "prompt": "Escribe un script en bash robusto que utilice rsync para realizar backups incrementales de un directorio de origen a un destino, rotando logs diarios y manejando errores de conexión.",
    },
    "2_algoritmia_compleja": {
        "categoria": "Algoritmia Avanzada",
        "system": "Eres un experto en ciencias de la computación.",
        "prompt": "Implementa en Python el algoritmo A* (A-star) para encontrar el camino más corto en una cuadrícula 2D. Incluye comentarios explicando el cálculo de la heurística.",
    },
    "3_resolucion_problemas": {
        "categoria": "Depuración y Seguridad",
        "system": "Eres un auditor de código especializado en seguridad y bugs escurridizos.",
        "prompt": "Revisa este código Python, identifica el bug de mutabilidad y reescríbelo correctamente:\n\ndef agregar_item(item, lista=[]):\n    lista.append(item)\n    return lista",
    },
    "4_uso_herramientas_agente": {
        "categoria": "Capacidad Agéntica (Tool Calling)",
        "system": "Eres un agente de IA. Debes usar llamadas a funciones. Responde ÚNICAMENTE con un objeto JSON válido que represente la llamada a la herramienta.",
        "prompt": "El usuario pide: 'Busca todos los archivos .conf en el directorio /etc/ que contengan la palabra puerto'. Genera el JSON para llamar a la función 'buscar_texto_en_archivos' con los argumentos 'directorio', 'extension', y 'texto_busqueda'.",
    },
    "5_refactorizacion_y_opt": {
        "categoria": "Optimización de Código",
        "system": "Eres un ingeniero de rendimiento de software. Tu meta es hacer el código lo más eficiente posible.",
        "prompt": "Optimiza la siguiente función de Python que busca duplicados. Actualmente es O(N^2), redúcela a O(N) en tiempo:\n\ndef encontrar_duplicados(lista):\n    duplicados = []\n    for i in range(len(lista)):\n        for j in range(i + 1, len(lista)):\n            if lista[i] == lista[j] and lista[i] not in duplicados:\n                duplicados.append(lista[i])\n    return duplicados",
    },
    "6_generacion_unit_tests": {
        "categoria": "Calidad y Testing",
        "system": "Eres un QA Automation Engineer experto en Python y pytest.",
        "prompt": "Escribe una suite completa de pruebas unitarias usando 'pytest' para una función ficticia `validar_password(password: str) -> bool`. Debes probar: longitud mínima de 8, presencia de un número, una mayúscula y manejo de strings vacíos.",
    },
    "7_sql_avanzado": {
        "categoria": "Bases de Datos",
        "system": "Eres un DBA y Data Engineer experto en PostgreSQL.",
        "prompt": "Dadas dos tablas: `empleados` (id, nombre, departamento_id, salario) y `departamentos` (id, nombre_depto). Escribe una consulta SQL optimizada que devuelva el nombre del departamento y el empleado con el salario más alto dentro de cada departamento (usa funciones de ventana/window functions).",
    },
    "8_api_resiliente_y_errores": {
        "categoria": "Integración y Resiliencia",
        "system": "Eres un desarrollador Backend experto en integraciones robustas.",
        "prompt": "Escribe una función en Python usando `requests` para consultar un endpoint HTTP GET. La función debe implementar un mecanismo de reintento (retry) con backoff exponencial (máximo 3 intentos) si el servidor responde con un error de la serie 5xx.",
    },
    "9_logica_razonamiento": {
        "categoria": "Razonamiento Lógico",
        "system": "Eres un experto en resolución de problemas lógicos. Explica tu razonamiento paso a paso.",
        "prompt": "Un hombre necesita cruzar un río con un zorro, un pollo y un grano de maíz. Solo puede llevar uno de ellos a la vez. Si deja al zorro con el pollo, el zorro se come al pollo; si deja al pollo con el grano, el pollo se come el grano. ¿Cómo logra cruzar a todos sanos y salvos?",
    },
    "10_matematicas_ecuacion": {
        "categoria": "Matemáticas",
        "system": "Eres un matemático. Resuelve el problema mostrando todos los pasos.",
        "prompt": "Resuelve la ecuación diferencial dy/dx = y. Proporciona la solución general y explica el método utilizado.",
    },
    "11_extraccion_info": {
        "categoria": "Procesamiento de Lenguaje Natural",
        "system": "Eres un sistema de extracción de información. Extrae entidades (personas, lugares, fechas, organizaciones) del siguiente texto y devuélvelas en formato JSON.",
        "prompt": "Texto: 'El 15 de mayo de 2023, la empresa OpenAI anunció desde San Francisco que su nuevo modelo GPT-4 sería presentado por Sam Altman durante la conferencia anual.'",
    },
    "12_generacion_go": {
        "categoria": "Generación Código (Go)",
        "system": "Eres un desarrollador Go. Responde solo con código.",
        "prompt": "Escribe un servidor HTTP en Go que escuche en el puerto 8080 y tenga un endpoint '/health' que devuelva un JSON con el estado 'ok'.",
    },
    "13_generacion_rust": {
        "categoria": "Generación Código (Rust)",
        "system": "Eres un desarrollador Rust. Responde solo con código.",
        "prompt": "Implementa una función en Rust que calcule el factorial de un número de forma recursiva y maneje el caso de números negativos devolviendo un Result.",
    },
    "14_generacion_js": {
        "categoria": "Generación Código (JavaScript)",
        "system": "Eres un desarrollador JavaScript moderno (ES6+). Responde solo con código.",
        "prompt": "Escribe una función asíncrona que realice una solicitud GET a una API, con reintentos en caso de fallo (máximo 3 intentos) usando fetch y manejo de errores.",
    },
    "15_resumen_texto": {
        "categoria": "Comprensión y Resumen",
        "system": "Eres un asistente especializado en resumir textos de forma clara y concisa.",
        "prompt": "Resume el siguiente texto en un párrafo de no más de 100 palabras:\n\n'La inteligencia artificial (IA) es un campo de la informática que se enfoca en la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana, como el aprendizaje, el razonamiento y la percepción. En los últimos años, los avances en aprendizaje profundo y redes neuronales han impulsado aplicaciones como el reconocimiento de voz, la visión por computadora y los vehículos autónomos. Sin embargo, la IA también plantea desafíos éticos, como el sesgo algorítmico, la privacidad de los datos y el impacto en el empleo. La comunidad científica aboga por un desarrollo responsable que garantice que la IA beneficie a la sociedad en su conjunto.'",
    },
    "16_traduccion": {
        "categoria": "Traducción",
        "system": "Eres un traductor experto. Traduce el siguiente texto del inglés al español manteniendo el tono técnico.",
        "prompt": "Translate to Spanish:\n'Machine learning is a subfield of artificial intelligence that gives computers the ability to learn without being explicitly programmed. It uses algorithms that iteratively learn from data.'",
    },
    "17_explicacion_concepto": {
        "categoria": "Explicación de Conceptos",
        "system": "Eres un profesor de ciencias de la computación. Explica conceptos de forma didáctica.",
        "prompt": "Explica qué es la notación Big O y da ejemplos de O(1), O(n), O(n^2) y O(log n).",
    },
    "18_tool_calling_avanzado": {
        "categoria": "Tool Calling Avanzado",
        "system": "Eres un agente que debe usar múltiples herramientas. Responde ÚNICAMENTE con un JSON que contenga una lista de llamadas a funciones.",
        "prompt": "El usuario quiere: 1) Obtener el clima actual en Madrid, 2) Calcular la raíz cuadrada de 256, 3) Enviar un correo a juan@example.com con el resultado. Genera el JSON para llamar a las funciones 'obtener_clima', 'calcular_raiz' y 'enviar_correo' con los argumentos adecuados.",
    },
    "19_seguridad_injection": {
        "categoria": "Seguridad y Análisis de Código",
        "system": "Eres un experto en seguridad ofensiva y defensiva.",
        "prompt": "El siguiente código PHP es vulnerable a inyección SQL. Identifica la vulnerabilidad y escribe una versión segura usando sentencias preparadas:\n\n$id = $_GET['id'];\n$query = \"SELECT * FROM usuarios WHERE id = $id\";\n$result = mysqli_query($conn, $query);",
    },
}


# ─── Obtener nombre del modelo ───────────────────────────────────────────────

def obtener_nombre_modelo(cfg: ServerConfig) -> str:
    try:
        resp = requests.get(f"{cfg.host}/props", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            nombre = data.get("model_alias") or data.get("model_path")
            if nombre:
                return nombre
    except Exception:
        pass

    try:
        resp = requests.get(f"{cfg.host}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0].get("id", "Modelo (no reportado)")
    except Exception:
        pass

    return "Modelo Desconocido"


# ─── Métricas Prometheus ─────────────────────────────────────────────────────

def obtener_metricas(cfg: ServerConfig) -> str:
    try:
        respuesta = requests.get(cfg.metrics_endpoint, timeout=2)
        respuesta.raise_for_status()
        return respuesta.text
    except Exception as e:
        return f"Error obteniendo métricas: {e}"


def parsear_metricas(raw_metrics: str) -> MetricsMap:
    datos: MetricsMap = {}
    patrones = [
        r'llamacpp:tokens_predicted_total\s+(\d+)',
        r'llamacpp:tokens_eval_total\s+(\d+)',
        r'llama\.cpp:tokens_predicted_total\s+(\d+)',
        r'llama\.cpp:tokens_eval_total\s+(\d+)',
        r'tokens_(?:predicted|generated|completion)_total\s+(\d+)',
        r'tokens_(?:eval|prompt)_total\s+(\d+)',
    ]

    for pat in patrones:
        match = re.search(pat, raw_metrics)
        if match:
            if "predicted" in pat or "generated" in pat or "completion" in pat:
                datos['tokens_generados_totales'] = int(match.group(1))
            elif "eval" in pat or "prompt" in pat:
                datos['tokens_prompt_totales'] = int(match.group(1))

    # Si no se encontraron métricas específicas, intentar sumar todas las que contengan "tokens"
    if not datos:
        tokens_total = 0
        for line in raw_metrics.splitlines():
            if "tokens" in line and "total" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    tokens_total += int(parts[1])
        if tokens_total:
            datos['tokens_generados_totales'] = tokens_total  # asumimos que son generados

    datos.setdefault('tokens_generados_totales', 0)
    datos.setdefault('tokens_prompt_totales', 0)
    return datos


# ─── Ejecución de pruebas ────────────────────────────────────────────────────

def ejecutar_prueba(nombre: str, datos_test: dict[str, str], cfg: ServerConfig) -> TestResult:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": datos_test["system"]},
            {"role": "user", "content": datos_test["prompt"]},
        ],
        "temperature": 0.1,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }

    inicio = time.monotonic()
    try:
        resp = requests.post(cfg.chat_endpoint, json=payload, timeout=cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        tiempo = round(time.monotonic() - inicio, 2)

        # Verificar que la respuesta tenga el formato esperado
        if 'choices' not in data or not data['choices']:
            raise ValueError("Respuesta del servidor sin 'choices'")
        contenido = data['choices'][0].get('message', {}).get('content', '')
        if not contenido:
            raise ValueError("Contenido vacío en la respuesta")

        usage = data.get('usage', {})
        prompt_tk = usage.get('prompt_tokens', 0)
        completion_tk = usage.get('completion_tokens', 0)
        tps = round(completion_tk / tiempo, 2) if tiempo > 0 else 0

        validador = VALIDATORS.get(nombre)
        if validador is not None:
            val_ok, val_msg, val_score = validador(contenido)
        else:
            val_ok = bool(contenido.strip())
            val_msg = "Respuesta no vacía (sin validador específico)." if val_ok else "Respuesta vacía."
            val_score = 1.0 if val_ok else 0.0

        return {
            "categoria": datos_test["categoria"],
            "tiempo_segundos": tiempo,
            "tps_generacion": tps,
            "prompt_tokens": prompt_tk,
            "gen_tokens": completion_tk,
            "respuesta": contenido,
            "valida_ok": val_ok,
            "valida_msg": val_msg,
            "valida_score": val_score,
            "model": data.get("model"),
            "error": None,
        }

    except Exception as e:
        return {
            "categoria": datos_test["categoria"],
            "error": str(e),
            "tiempo_segundos": round(time.monotonic() - inicio, 2),
            "tps_generacion": 0,
            "prompt_tokens": 0,
            "gen_tokens": 0,
            "respuesta": "",
            "valida_ok": False,
            "valida_msg": f"Error: {e}",
            "valida_score": 0.0,
            "model": None,
        }


# ─── Salida en consola (optimizada) ─────────────────────────────────────────

def print_header(texto: str) -> None:
    print()
    print(c(Color.CYAN + Color.BOLD, "=" * 70))
    print(c(Color.CYAN + Color.BOLD, f"  {texto}"))
    print(c(Color.CYAN + Color.BOLD, "=" * 70))


def print_test_progreso(indice: int, total: int, nombre: str, categoria: str, quiet: bool = False) -> None:
    if quiet:
        return
    prefijo = c(Color.BLUE, f"[{indice:02d}/{total:02d}]")
    # Usar \r para sobreescribir la línea
    sys.stdout.write(f"\r  {prefijo} {c(Color.BOLD, categoria)} -> {c(Color.GRAY, nombre)} ... ")
    sys.stdout.flush()


def print_test_resultado(res: TestResult, quiet: bool = False) -> None:
    if quiet:
        return
    if res.get("error"):
        print(f"{c(Color.RED, 'ERROR')} {c(Color.RED, str(res['error'][:60]))}")
        return

    tps = res.get("tps_generacion", 0)
    tiempo = res.get("tiempo_segundos", 0)
    pt = res.get("prompt_tokens", 0)
    gt = res.get("gen_tokens", 0)
    score = res.get("valida_score", 0.0)
    val_msg = res.get("valida_msg", "")

    # Acortar mensaje de validación si es muy largo
    if len(val_msg) > 40:
        val_msg = val_msg[:37] + "..."

    match (res.get("valida_ok"), score >= 0.5):
        case (True, _):
            icono = c(Color.GREEN, "OK")
        case (False, True):
            icono = c(Color.YELLOW, "PARCIAL")
        case _:
            icono = c(Color.RED, "FAIL")

    # Mostrar solo el resultado, sin repetir la línea de progreso
    print(f" {icono} {tiempo:.1f}s | {tps:.1f} TPS | {pt}/{gt} tk | {val_msg[:50]}")


def print_tabla_resumen(resultados: ResultsMap) -> None:
    print()
    print(c(Color.BOLD + Color.CYAN, "  TEST                                 ESTADO   TIEMPO    TPS      SCORE"))
    print(c(Color.CYAN, "  " + "-" * 66))

    for nombre, res in resultados.items():
        cat = res.get("categoria", "")
        nombre_corto = cat[:35].ljust(35)

        if res.get("error"):
            estado = c(Color.RED, "ERROR ")
            tiempo_str = f"{res['tiempo_segundos']:.1f}s"
            tps_str = "---"
            score_str = "  0.0"
        else:
            score = res.get("valida_score", 0.0)
            ok = res.get("valida_ok", False)
            if ok and score == 1.0:
                estado = c(Color.GREEN, " OK   ")
            elif ok or score >= 0.5:
                estado = c(Color.YELLOW, "PARCIAL")
            else:
                estado = c(Color.RED, "FAIL  ")

            tiempo_str = f"{res['tiempo_segundos']:.1f}s"
            tps_str = f"{res['tps_generacion']:.1f}" if res.get("tps_generacion", 0) > 0 else "---"
            score_str = f"  {score:.1f}"

        print(f"  {nombre_corto} {estado} {tiempo_str:>7}  {tps_str:>7}  {score_str}")


def print_resumen_final(
    nombre_modelo: str,
    resultados: ResultsMap,
    total_pt: int,
    total_gt: int,
    tiempo_total: float,
    tps_promedio: float,
    tps_lista: list[float],
) -> None:
    total = len(resultados)
    exitos = sum(
        1 for r in resultados.values()
        if r.get("valida_ok") and r.get("valida_score", 0) == 1.0 and not r.get("error")
    )
    parciales = sum(
        1 for r in resultados.values()
        if not r.get("error")
        and r.get("valida_score", 0) >= 0.5
        and not (r.get("valida_ok") and r.get("valida_score", 0) == 1.0)
    )
    errores = sum(1 for r in resultados.values() if r.get("error"))
    fallos = total - exitos - parciales - errores
    porcentaje = (exitos / total * 100) if total > 0 else 0

    print()
    print(c(Color.BOLD + Color.CYAN, "=" * 70))
    print(c(Color.BOLD + Color.CYAN, "  RESUMEN DE EVALUACION"))
    print(c(Color.CYAN, "=" * 70))
    print()
    print(f"  {c(Color.BOLD, 'Modelo:')}        {nombre_modelo}")
    print(f"  {c(Color.BOLD, 'Fecha:')}         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print(c(Color.BOLD, "  RESULTADOS:"))
    print(f"    {c(Color.GREEN, 'Exitosos:')}    {exitos}/{total} ({porcentaje:.1f}%)")
    if parciales > 0:
        print(f"    {c(Color.YELLOW, 'Parciales:')}   {parciales}/{total}")
    if fallos > 0:
        print(f"    {c(Color.RED, 'Fallidos:')}    {fallos}/{total}")
    if errores > 0:
        print(f"    {c(Color.RED, 'Errores:')}     {errores}/{total}")
    print()

    print(c(Color.BOLD, "  RENDIMIENTO:"))
    print(f"    {c(Color.BOLD, 'Tokens entrada:')}   {total_pt:,}")
    print(f"    {c(Color.BOLD, 'Tokens salida:')}    {total_gt:,}")
    print(f"    {c(Color.BOLD, 'Tokens total:')}     {total_pt + total_gt:,}")
    print(f"    {c(Color.BOLD, 'Tiempo total:')}     {tiempo_total:.2f}s")

    if tps_lista:
        print(f"    {c(Color.BOLD, 'TPS promedio:')}     {tps_promedio:.2f} tok/s")
        print(f"    {c(Color.BOLD, 'TPS min/max:')}      {min(tps_lista):.1f} / {max(tps_lista):.1f}")
        if len(tps_lista) > 1:
            print(f"    {c(Color.BOLD, 'Desv. estandar:')}  {statistics.stdev(tps_lista):.2f}")
    print()

    # Barra de progreso visual
    barra_len = 40
    llenos = int((exitos / total) * barra_len) if total > 0 else 0
    barra = c(Color.GREEN, "█" * llenos) + c(Color.GRAY, "░" * (barra_len - llenos))
    print(f"  [{barra}] {porcentaje:.1f}%")
    print()


# ─── Construcción de informe ─────────────────────────────────────────────────

def _calcular_estadisticas(resultados: ResultsMap) -> dict[str, int]:
    """Calcula estadísticas de resultados de forma funcional."""
    exitos = sum(
        1 for r in resultados.values()
        if r.get("valida_ok") and r.get("valida_score", 0) == 1.0 and not r.get("error")
    )
    parciales = sum(
        1 for r in resultados.values()
        if not r.get("error")
        and r.get("valida_score", 0) >= 0.5
        and not (r.get("valida_ok") and r.get("valida_score", 0) == 1.0)
    )
    errores = sum(1 for r in resultados.values() if r.get("error"))
    return {"exitos": exitos, "parciales": parciales, "errores": errores}


def _construir_informe(
    nombre_modelo: str,
    resultados: ResultsMap,
    cfg: ServerConfig,
    stats: dict[str, int],
    total_pt: int,
    total_gt: int,
    tiempo_total: float,
    tps_promedio: float,
    tps_lista: list[float],
    interrumpido: bool,
    metricas: MetricsMap,
) -> dict[str, Any]:
    """Construye el diccionario del informe final."""
    total = len(resultados)

    return {
        "metadata": {
            "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "host": cfg.host,
            "timeout": cfg.timeout,
            "max_tokens": cfg.max_tokens,
            "nombre_modelo": nombre_modelo or "Desconocido",
            "total_pruebas": total,
            "interrumpido": interrumpido,
        },
        "resumen": {
            "pruebas_exitosas": stats["exitos"],
            "pruebas_parciales": stats["parciales"],
            "pruebas_fallidas": total - stats["exitos"] - stats["parciales"] - stats["errores"],
            "pruebas_con_error": stats["errores"],
            "porcentaje_exito": f"{stats['exitos'] / total * 100:.1f}%" if total > 0 else "0%",
            "tiempo_total": round(tiempo_total, 2),
            "tps_promedio": tps_promedio,
            "tps_min": round(min(tps_lista), 1) if tps_lista else 0,
            "tps_max": round(max(tps_lista), 1) if tps_lista else 0,
            "tokens_totales": total_pt + total_gt,
        },
        "estado_servidor": metricas,
        "resultados_pruebas": resultados,
    }


# ─── Función principal ───────────────────────────────────────────────────────

def evaluar_modelo(cfg: ServerConfig) -> None:
    resultados_informe: ResultsMap = {}
    nombre_modelo: str | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tiempo_total = 0.0
    tps_lista: list[float] = []

    if not cfg.quiet:
        print_header("EVALUACION DE MODELO EN LLAMA.CPP")

    nombre_modelo = obtener_nombre_modelo(cfg)
    if not cfg.quiet:
        print(f"\n  {c(Color.BOLD, 'Modelo detectado:')} {c(Color.CYAN, nombre_modelo)}")
        print(f"  {c(Color.BOLD, 'Host:')} {cfg.host}")
        print(f"  {c(Color.BOLD, 'Timeout:')} {cfg.timeout}s | {c(Color.BOLD, 'Max tokens:')} {cfg.max_tokens}")
        print()

    # Filtrar tests si se especificó --tests (soporte para rangos, ej. "1-5,7")
    items: list[tuple[str, dict[str, str]]] = list(TEST_SUITE.items())
    if cfg.tests_filter:
        test_ids = []
        for part in cfg.tests_filter.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                test_ids.extend(range(int(start), int(end)+1))
            else:
                test_ids.append(int(part))
        items = [(k, v) for i, (k, v) in enumerate(items, 1) if i in test_ids]

    total_tests = len(items)

    # Handler para Ctrl+C
    interrumpido = False

    def signal_handler(sig: int, frame: Any) -> None:
        nonlocal interrumpido
        interrumpido = True
        if not cfg.quiet:
            print(f"\n\n{c(Color.YELLOW, ' Interrupcion detectada. Guardando resultados parciales...')}")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for i, (nombre_test, datos_test) in enumerate(items, 1):
            if interrumpido:
                break
            if not cfg.quiet:
                print_test_progreso(i, total_tests, nombre_test, datos_test["categoria"])
            res = ejecutar_prueba(nombre_test, datos_test, cfg)
            resultados_informe[nombre_test] = res
            if not cfg.quiet:
                print_test_resultado(res)

            if (not nombre_modelo or "Desconocido" in nombre_modelo) and not res.get("error") and res.get("model"):
                nombre_modelo = res.get("model")

            if not res.get("error"):
                total_prompt_tokens += res.get("prompt_tokens", 0)
                total_completion_tokens += res.get("gen_tokens", 0)
                tiempo_total += res.get("tiempo_segundos", 0)
                tps = res.get("tps_generacion", 0)
                if tps > 0:
                    tps_lista.append(tps)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Si se interrumpió, añadir una nueva línea para separar
    if interrumpido and not cfg.quiet:
        print()

    # Métricas del servidor
    metricas_raw = obtener_metricas(cfg)
    metricas_parseadas = parsear_metricas(metricas_raw)

    stats = _calcular_estadisticas(resultados_informe)
    tps_promedio = round(sum(tps_lista) / len(tps_lista), 2) if tps_lista else 0

    # Resumen en consola
    if not cfg.quiet:
        print_tabla_resumen(resultados_informe)
    print_resumen_final(
        nombre_modelo, resultados_informe,
        total_prompt_tokens, total_completion_tokens,
        tiempo_total, tps_promedio, tps_lista,
    )

    # Construir informe
    modelo_sanitizado = sanitizar_nombre(nombre_modelo)
    informe_final = _construir_informe(
        nombre_modelo, resultados_informe, cfg, stats,
        total_prompt_tokens, total_completion_tokens,
        tiempo_total, tps_promedio, tps_lista,
        interrumpido, metricas_parseadas,
    )

    # Guardar JSON con pathlib
    filename_json = Path(f"reporte_{modelo_sanitizado}.json")
    if filename_json.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_json = Path(f"reporte_{modelo_sanitizado}_{suffix}.json")

    filename_json.write_text(
        json.dumps(informe_final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not cfg.quiet:
        print(c(Color.BOLD, f"  JSON guardado: {c(Color.CYAN, str(filename_json))}"))
        print(c(Color.CYAN, "=" * 70))
        print()

    # Opcional: guardar log a archivo si se especificó
    if cfg.log_file:
        log_path = Path(cfg.log_file)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - Evaluación completada para {nombre_modelo}\n")
            f.write(f"  Exitosos: {stats['exitos']}/{len(resultados_informe)}\n")
            f.write(f"  TPS promedio: {tps_promedio:.2f}\n")
            f.write(f"  Tiempo total: {tiempo_total:.2f}s\n\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos de IA con llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.getenv("LLAMA_HOST", "http://127.0.0.1:8080"),
        help="Host del servidor llama.cpp (default: LLAMA_HOST o http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TIMEOUT", "120")),
        help="Timeout por petición en segundos (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "4096")),
        help="Máximo de tokens de salida (default: 4096)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="IDs de tests a ejecutar, separados por coma; soporta rangos (ej: 1,3-5,7)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime la salida detallada (solo muestra el resumen final).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Archivo de log para registrar resultados (append).",
    )

    args = parser.parse_args()

    return ServerConfig(
        host=args.host,
        chat_endpoint=f"{args.host}/v1/chat/completions",
        metrics_endpoint=f"{args.host}/metrics",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        tests_filter=args.tests,
        quiet=args.quiet,
        log_file=args.log,
    )


def main() -> None:
    cfg = parse_args()
    evaluar_modelo(cfg)


if __name__ == "__main__":
    main()