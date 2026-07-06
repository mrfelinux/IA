"""Shared fixtures for TEST-IA unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_json_response() -> dict:
    """A dict with nested data simulating a tool-call response."""
    return {
        "metadata": {
            "nombre_modelo": "llama-3.2-3b-instr",
            "fecha_ejecucion": "2026-07-06T10:00:00",
            "version": "0.1.0",
        },
        "resumen": {
            "pruebas_exitosas": 15,
            "pruebas_parciales": 3,
            "pruebas_fallidas": 1,
            "porcentaje_exito": "78.9%",
            "tps_promedio": 45.2,
            "tiempo_total": 320,
            "tokens_totales": 14500,
        },
        "resultados_pruebas": {
            "1_generacion_codigo": {
                "categoria": "Generación de Código",
                "valida_score": 1.0,
                "valida_msg": "Script bash completo",
            },
            "2_algoritmia_compleja": {
                "categoria": "Algoritmia Avanzada",
                "valida_score": 0.83,
                "valida_msg": "Implementación A* plausible",
            },
        },
    }


@pytest.fixture
def sample_markdown_with_code() -> str:
    """A string with a python code block in markdown."""
    return (
        'Para calcular la suma, usa:\n\n'
        '```python\n'
        'def suma(a: int, b: int) -> int:\n'
        '    return a + b\n'
        '```\n\n'
        'Esto devuelve la suma de dos números.'
    )


@pytest.fixture
def sample_markdown_with_bash() -> str:
    """A string with a bash code block in markdown."""
    return (
        'Para hacer un backup:\n\n'
        '```bash\n'
        'rsync -avz /origen/ /destino/\n'
        'echo "Backup completado"\n'
        '```\n\n'
        'Esto sincroniza los directorios.'
    )


@pytest.fixture
def sample_json_markdown() -> str:
    """A string with a json code block in markdown."""
    return (
        'La respuesta es:\n\n'
        '```json\n'
        '{"name": "get_weather", "arguments": {"city": "Madrid"}}\n'
        '```\n\n'
        'Fin de la respuesta.'
    )


@pytest.fixture
def deeply_nested_preamble() -> str:
    """Text with a fake ``{a}`` before the real JSON — regression guard."""
    return 'prefix {a} then { "key": "value" }'
