#!/usr/bin/env python3
"""Prueba básica de tool calling con llama.cpp."""
import json
import sys
from typing import Any

import requests

from tools_defs import TOOLS

SERVER = "http://localhost:8080/v1"

# ---------- Casos de prueba ----------
TEST_CASES = [
    {
        "description": "Clima de Madrid",
        "messages": [{"role": "user", "content": "¿Qué tiempo hace ahora en Madrid?"}],
        "expected_tool": "get_weather",
        "expected_args": {"city": "Madrid"}  # al menos city debe estar presente
    },
    {
        "description": "Búsqueda web",
        "messages": [{"role": "user", "content": "Busca información sobre la fotosíntesis"}],
        "expected_tool": "search_web",
        "expected_args": {"query": "fotosíntesis"}  # debe contener la palabra clave
    },
    {
        "description": "Enviar correo",
        "messages": [{"role": "user", "content": "Envía un email a juan@example.com con asunto 'Hola' y cuerpo 'Saludos'"}],
        "expected_tool": "send_email",
        "expected_args": {"to": "juan@example.com", "subject": "Hola", "body": "Saludos"}
    },
    {
        "description": "Ejecutar código",
        "messages": [{"role": "user", "content": "Ejecuta: print(2+2)"}],
        "expected_tool": "run_python_code",
        "expected_args": {"code": "print(2+2)"}
    }
]

# ---------- Llamada al API ----------
def chat_completion(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: str = "auto",
) -> dict[str, Any] | None:
    payload = {
        "model": "cualquiera",
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
    }
    try:
        resp = requests.post(f"{SERVER}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Error en la llamada: {e}")
        return None

# ---------- Verificación ----------
def check_tool_call(case: dict[str, Any], response: dict[str, Any] | None) -> tuple[bool, str]:
    """Compara la respuesta real con la esperada. Devuelve (éxito, detalle)."""
    if not response:
        return False, "Sin respuesta"
    choice = response.get("choices", [{}])[0]
    finish = choice.get("finish_reason")
    if finish != "tool_calls":
        return False, f"finish_reason={finish}, esperado 'tool_calls'. Mensaje: {choice.get('message',{}).get('content','')}"
    tool_calls = choice.get("message", {}).get("tool_calls", [])
    if not tool_calls:
        return False, "No hay tool_calls en el mensaje"
    if len(tool_calls) != 1:
        names = [c.get("function", {}).get("name", "desconocida") for c in tool_calls]
        return False, f"Se esperaba una sola tool_call, recibidas {len(tool_calls)}: {names}"
    call = tool_calls[0]  # solo evaluamos la primera
    func = call.get("function", {})
    name = func.get("name", "")
    args_str = func.get("arguments", "{}")
    # Parsear JSON
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return False, f"JSON de argumentos inválido: {args_str}"
    
    # Comprobar nombre
    expected_name = case["expected_tool"]
    if name != expected_name:
        return False, f"Herramienta incorrecta: '{name}' (esperada '{expected_name}')"
    
    # Comprobar argumentos (al menos las claves esperadas con valores aproximados)
    expected_args = case["expected_args"]
    for key, expected_val in expected_args.items():
        if key not in args:
            return False, f"Falta argumento '{key}' en {args}"
        actual_val = args[key]
        # Comparación flexible (cadena contiene valor esperado)
        if isinstance(expected_val, str):
            actual_str = str(actual_val).strip()
            expected_str = expected_val.strip()
            strict_keys = {"to", "subject", "body", "code"}
            strict_tools = {"send_email", "run_python_code"}
            if key in strict_keys or name in strict_tools:
                if actual_str != expected_str:
                    return False, f"Valor de '{key}': '{actual_val}' != '{expected_val}'"
            elif expected_str.lower() not in actual_str.lower():
                return False, f"Valor de '{key}': '{actual_val}' no contiene '{expected_val}'"
        elif not isinstance(expected_val, str) and actual_val != expected_val:
            return False, f"Valor de '{key}': {actual_val} != {expected_val}"
    
    return True, f"Correcto → {name}({args})"

# ---------- Ejecutar pruebas ----------
def run_tests() -> None:
    print("🔧 Probando herramientas con llama.cpp\n")
    success = 0
    for case in TEST_CASES:
        print(f"--- {case['description']} ---")
        resp = chat_completion(case["messages"], TOOLS)
        ok, detail = check_tool_call(case, resp)
        status = "✅" if ok else "❌"
        print(f"  {status} {detail}")
        if ok:
            success += 1
        print()
    print(f"Resultado: {success}/{len(TEST_CASES)} pruebas correctas")

# ---------- Extras: concurrencia y mezcla de herramientas ----------
def test_multiple_tools_same_prompt() -> None:
    """Envía un prompt que podría requerir varias herramientas (modelo debe elegir una)."""
    print("🔀 Prueba de ambigüedad (debe elegir la más adecuada)")
    messages = [{"role": "user", "content": "Busca en internet el clima de París"}]
    resp = chat_completion(messages, TOOLS)
    if resp:
        choice = resp["choices"][0]
        if choice["finish_reason"] == "tool_calls":
            call = choice["message"]["tool_calls"][0]
            print(f"  Herramienta elegida: {call['function']['name']}")
            print(f"  Argumentos: {call['function']['arguments']}")
        else:
            print("  El modelo no llamó a ninguna herramienta.")
    else:
        print("  Falló la petición")

if __name__ == "__main__":
    run_tests()
    test_multiple_tools_same_prompt()
