"""Tests for validation/helper functions from ia-test.py.

Functions tested:
  - calcular_score       — score from two bools (1.0 / 0.5 / 0.0)
  - sanitizar_nombre     — sanitise model names for filenames
  - _codigo_python_parseado — extract and AST-parse python code
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

# ─── Dynamic import for ia-test.py (hyphen in filename) ───────────────────────
_TEST_IA_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ia_test_source", str(_TEST_IA_DIR / "ia-test.py")
)
_ia_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ia_test)

calcular_score = _ia_test.calcular_score
sanitizar_nombre = _ia_test.sanitizar_nombre
_codigo_python_parseado = _ia_test._codigo_python_parseado


# ─── calcular_score ───────────────────────────────────────────────────────────


class TestCalcularScore:
    def test_both_true(self):
        """Both parts True → 1.0."""
        assert calcular_score(True, True) == 1.0

    def test_only_a_true(self):
        """Only part_a True → 0.5."""
        assert calcular_score(True, False) == 0.5

    def test_only_b_true(self):
        """Only part_b True → 0.5."""
        assert calcular_score(False, True) == 0.5

    def test_both_false(self):
        """Both parts False → 0.0."""
        assert calcular_score(False, False) == 0.0


# ─── sanitizar_nombre ─────────────────────────────────────────────────────────


class TestSanitizarNombre:
    def test_normal_name(self):
        """A normal name: spaces become underscores, dots stripped."""
        result = sanitizar_nombre("Llama 3.2 3B")
        assert result == "Llama_32_3B"

    def test_special_chars(self):
        """Special characters are stripped."""
        result = sanitizar_nombre("model@#$%^&*()_test")
        assert result == "model_test"

    def test_unicode(self):
        """Unicode characters are preserved (only non-word stripped)."""
        result = sanitizar_nombre("模型名称-v2")
        # Chinese characters are word chars in regex, hyphen is kept by \w
        assert "模型名称" in result or "模型" in result

    def test_empty_string(self):
        """Empty input returns the fallback name."""
        assert sanitizar_nombre("") == "modelo_desconocido"

    def test_whitespace_only(self):
        """Whitespace-only input returns the fallback name."""
        assert sanitizar_nombre("   ") == "modelo_desconocido"

    def test_very_long_name(self):
        """Names longer than 80 chars are truncated."""
        long_name = "A" * 200
        result = sanitizar_nombre(long_name)
        assert len(result) <= 80

    def test_trailing_underscores_stripped(self):
        """Trailing underscores after substitution are removed."""
        result = sanitizar_nombre("nombre_")
        assert result == "nombre"


# ─── _codigo_python_parseado ──────────────────────────────────────────────────


class TestCodigoPythonParseado:
    def test_valid_python(self):
        """Returns (code, ast.Module, None) for valid Python."""
        codigo, tree, error = _codigo_python_parseado("```python\nx = 1\n```")
        assert codigo == "x = 1"
        assert isinstance(tree, ast.Module)
        assert error is None

    def test_syntax_error(self):
        """Returns (code, None, error_msg) for syntactically invalid code."""
        codigo, tree, error = _codigo_python_parseado("```python\nx = 1\n invalid syntax$$\n```")
        assert codigo == "x = 1\n invalid syntax$$"
        assert tree is None
        assert error is not None
        assert "Error de sintaxis" in error

    def test_no_code(self):
        """Returns (empty_str, None, msg) when no code block is found."""
        codigo, tree, error = _codigo_python_parseado("Solo texto, sin código.")
        assert codigo == "Solo texto, sin código."
        # When there's no ``` block, extraer_codigo_python returns the whole text
        # and _codigo_python_parseado tries to AST-parse it
        # "Solo texto" is not valid Python, so it should return a syntax error
        assert tree is None
        assert error is not None

    def test_valid_python_uses_ast_parse(self):
        """Verify the function uses ast.parse internally for validation."""
        _, tree, error = _codigo_python_parseado("```python\nimport os\nprint(os.getpid())\n```")
        assert isinstance(tree, ast.Module)
        assert error is None
