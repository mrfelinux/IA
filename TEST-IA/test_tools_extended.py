#!/usr/bin/env python3
"""
Prueba de tool calling con llama.cpp.
Evalúa si el modelo elige correctamente entre múltiples herramientas y genera argumentos JSON válidos.
"""
import requests
import json
import sys
from typing import Dict, List, Optional

SERVER = "http://localhost:8080/v1"

# ---------- Definición de herramientas (7) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtiene el clima actual de una ciudad",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca información en Internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Envía un correo electrónico",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Ejecuta código Python y devuelve el resultado",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un archivo del sistema",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe contenido en un archivo",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Ejecuta una consulta SQL en la base de datos",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "database": {"type": "string", "default": "default"}
                },
                "required": ["query"]
            }
        }
    }
]

# ---------- Casos de prueba ----------
TEST_CASES = [
    # Herramienta: get_weather
    {
        "desc": "Clima en Madrid",
        "messages": [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}],
        "expected_tool": "get_weather",
        "expected_args_subset": {"city": "Madrid"}
    },
    # Herramienta: search_web
    {
        "desc": "Búsqueda web sobre fotosíntesis",
        "messages": [{"role": "user", "content": "Busca información sobre la fotosíntesis"}],
        "expected_tool": "search_web",
        "expected_args_subset": {"query": "fotosíntesis"}
    },
    # Herramienta: send_email
    {
        "desc": "Enviar correo",
        "messages": [{"role": "user", "content": "Envía un email a juan@example.com con asunto 'Hola' y cuerpo 'Saludos cordiales'"}],
        "expected_tool": "send_email",
        "expected_args_subset": {"to": "juan@example.com", "subject": "Hola", "body": "Saludos cordiales"}
    },
    # Herramienta: run_python_code
    {
        "desc": "Ejecutar código Python",
        "messages": [{"role": "user", "content": "Ejecuta este código: print('Hola mundo')"}],
        "expected_tool": "run_python_code",
        "expected_args_subset": {"code": "print('Hola mundo')"}
    },
    # Herramienta: read_file
    {
        "desc": "Leer archivo",
        "messages": [{"role": "user", "content": "Muéstrame el contenido de /etc/hosts"}],
        "expected_tool": "read_file",
        "expected_args_subset": {"path": "/etc/hosts"}
    },
    # Herramienta: write_file
    {
        "desc": "Escribir archivo",
        "messages": [{"role": "user", "content": "Guarda 'Hola mundo' en /tmp/saludo.txt"}],
        "expected_tool": "write_file",
        "expected_args_subset": {"path": "/tmp/saludo.txt", "content": "Hola mundo"}
    },
    # Herramienta: query_database
    {
        "desc": "Consulta SQL",
        "messages": [{"role": "user", "content": "Ejecuta esta consulta: SELECT * FROM usuarios"}],
        "expected_tool": "query_database",
        "expected_args_subset": {"query": "SELECT * FROM usuarios"}
    },
    # Ambigüedad: clima + búsqueda (debe preferir get_weather)
    {
        "desc": "Buscar clima en París (ambigüedad clima/búsqueda)",
        "messages": [{"role": "user", "content": "Busca en Internet el clima de París"}],
        "expected_tool": "get_weather",  # esperamos que priorice clima sobre búsqueda
        "expected_args_subset": {"city": "París"}
    }
]

# ---------- Función de llamada ----------
def chat_completion(messages, tools, tool_choice="auto", temperature=0):
    payload = {
        "model": "cualquiera",
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature
    }
    try:
        resp = requests.post(f"{SERVER}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Error HTTP: {e}")
        return None

# ---------- Evaluación ----------
def evaluate(case, response):
    if not response:
        return False, "Sin respuesta"
    choice = response.get("choices", [{}])[0]
    finish = choice.get("finish_reason")
    if finish != "tool_calls":
        msg = choice.get("message", {}).get("content", "")
        return False, f"finish_reason={finish}, contenido: {msg[:100]}"
    
    tool_calls = choice["message"].get("tool_calls", [])
    if not tool_calls:
        return False, "No se encontraron tool_calls"
    call = tool_calls[0]
    func = call.get("function", {})
    name = func.get("name")
    args_str = func.get("arguments", "{}")
    
    # Parseo JSON de argumentos
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return False, f"JSON inválido: {args_str}"
    
    expected_name = case["expected_tool"]
    if name != expected_name:
        return False, f"Herramienta incorrecta: '{name}' (esperada '{expected_name}')"
    
    # Verificar argumentos obligatorios
    for key, expected_val in case["expected_args_subset"].items():
        if key not in args:
            return False, f"Falta argumento '{key}'"
        actual = args[key]
        # Comprobación flexible para cadenas
        if isinstance(expected_val, str) and expected_val.lower() not in str(actual).lower():
            return False, f"Argumento '{key}': '{actual}' no contiene '{expected_val}'"
        elif not isinstance(expected_val, str) and actual != expected_val:
            return False, f"Argumento '{key}': {actual} != {expected_val}"
    return True, f"OK → {name}({args})"

# ---------- Ejecutar todas las pruebas ----------
def run_all():
    print("🔧 Prueba de tool calling con 7 herramientas\n")
    passed = 0
    for case in TEST_CASES:
        print(f"--- {case['desc']} ---")
        resp = chat_completion(case["messages"], TOOLS)
        ok, msg = evaluate(case, resp)
        print(f"  {'✅' if ok else '❌'} {msg}")
        if ok:
            passed += 1
        print()
    print(f"📊 Resultado: {passed}/{len(TEST_CASES)} pruebas correctas")

if __name__ == "__main__":
    run_all()
