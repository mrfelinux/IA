"""Tests for extraction functions from ia-test.py.

Functions tested:
  - extraer_codigo_python  — extract python code from markdown
  - extraer_codigo_bash    — extract bash/sh code from markdown
  - extraer_json           — extract JSON from markdown or raw text
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# ─── Dynamic import for ia-test.py (hyphen in filename) ───────────────────────
_TEST_IA_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ia_test_source", str(_TEST_IA_DIR / "ia-test.py")
)
assert _spec is not None and _spec.loader is not None
_ia_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ia_test)

extraer_codigo_python = _ia_test.extraer_codigo_python
extraer_codigo_bash = _ia_test.extraer_codigo_bash
extraer_json = _ia_test.extraer_json


# ─── extraer_codigo_python ────────────────────────────────────────────────────


class TestExtraerCodigoPython:
    def test_normal_block(self):
        """Extract code from a well-formed ```python block."""
        md = (
            "```python\n"
            'print("hello")\n'
            "```"
        )
        assert extraer_codigo_python(md) == 'print("hello")'

    def test_no_language_tag(self):
        """Fallback: extract from a plain ``` block when no language tag."""
        md = (
            "```\n"
            "print('fallback')\n"
            "```"
        )
        assert extraer_codigo_python(md) == "print('fallback')"

    def test_no_code_at_all(self):
        """When there is no code block, return the whole text."""
        text = "Esto es solo texto sin bloques de código."
        assert extraer_codigo_python(text) == text

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extraer_codigo_python("") == ""

    def test_unicode_content(self):
        """Code blocks with unicode characters are extracted correctly."""
        md = (
            "```python\n"
            "def saludar(nombre: str) -> str:\n"
            '    return f"Hola, {nombre}! 😊"\n'
            "```"
        )
        result = extraer_codigo_python(md)
        expected = 'def saludar(nombre: str) -> str:\n    return f"Hola, {nombre}! 😊"'
        assert result == expected

    def test_code_with_extra_text_around(self):
        """Extra text before/after the block is stripped, only code returned."""
        md = (
            "Aquí tienes el código:\n\n"
            "```python\n"
            "x = 42\n"
            "```\n\n"
            "Espero que te sirva."
        )
        assert extraer_codigo_python(md) == "x = 42"


# ─── extraer_codigo_bash ──────────────────────────────────────────────────────


class TestExtraerCodigoBash:
    def test_bash_block(self):
        """Extract from a ```bash block."""
        md = (
            "```bash\n"
            "ls -la /tmp\n"
            "```"
        )
        assert extraer_codigo_bash(md) == "ls -la /tmp"

    def test_sh_block(self):
        """Extract from a ```sh block (alternative tag)."""
        md = (
            "```sh\n"
            "echo 'hello'\n"
            "```"
        )
        assert extraer_codigo_bash(md) == "echo 'hello'"

    def test_no_block(self):
        """When there is no bash/sh block, return the whole text."""
        text = "Solo texto, ningún bloque de código."
        assert extraer_codigo_bash(text) == text

    def test_bash_block_without_tag(self):
        """A plain ``` block with bash-like content is still returned."""
        md = (
            "```\n"
            "rsync -av\n"
            "```"
        )
        assert extraer_codigo_bash(md) == "rsync -av"


# ─── extraer_json ─────────────────────────────────────────────────────────────


class TestExtraerJson:
    def test_json_block(self):
        """Extract JSON from a ```json code block."""
        md = (
            "```json\n"
            '{"city": "Madrid", "temp": 22}\n'
            "```"
        )
        assert extraer_json(md) == '{"city": "Madrid", "temp": 22}'

    def test_raw_json_string(self):
        """When the entire response is valid JSON, return it as-is."""
        raw = '{"name": "test", "value": 42}'
        assert extraer_json(raw) == raw

    def test_deeply_nested_preamble(self):
        """
        Regression: prefix with fake ``{a}`` before the real JSON.
        The function must skip the invalid ``{a}`` and find the real object.
        """
        text = 'prefix {a} then { "key": "value" }'
        result = extraer_json(text)
        # The extractor finds the real JSON object preserving original spacing
        assert "key" in result
        assert "value" in result
        assert json.loads(result) == {"key": "value"}

    def test_array_json(self):
        """Extract a JSON array from raw text."""
        text = 'Los resultados son: ["a", "b", "c"] según el análisis.'
        result = extraer_json(text)
        assert result == '["a", "b", "c"]'

    def test_malformed_json_returns_raw_text(self):
        """
        When no valid JSON is found anywhere in the text, return the
        entire response stripped.
        """
        text = "Esto no es JSON ni contiene JSON."
        assert extraer_json(text) == text

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extraer_json("") == ""

    def test_json_with_extra_text(self):
        """Extract JSON object embedded in surrounding text."""
        text = (
            "El resultado es:\n"
            '{"function": {"name": "get_weather", "arguments": {"city": "Madrid"}}}\n'
            "Fin."
        )
        result = extraer_json(text)
        assert '"get_weather"' in result
        assert '"Madrid"' in result
