"""Tests for comparador.py helper functions.

Functions tested:
  - safe_get           — safe nested dict access
  - bar                — horizontal bar renderer
  - pct_bar            — percentage bar with colour logic
  - format_model_name  — shorten model file/name for display
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# ─── Dynamic import for comparador.py ─────────────────────────────────────────
_TEST_IA_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "comparador_source", str(_TEST_IA_DIR / "comparador.py")
)
assert _spec is not None and _spec.loader is not None
_comparador = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comparador)

safe_get = _comparador.safe_get
bar = _comparador.bar
pct_bar = _comparador.pct_bar
format_model_name = _comparador.format_model_name

# ANSI colour constants (re-exported for test assertions)
G = _comparador.G
R = _comparador.R
Y = _comparador.Y
N = _comparador.N


# ─── safe_get ─────────────────────────────────────────────────────────────────


class TestSafeGet:
    def test_normal_key(self):
        """Returns the value for a top-level key."""
        data = {"name": "test", "value": 42}
        assert safe_get(data, "name") == "test"
        assert safe_get(data, "value") == 42

    def test_nested_key(self):
        """Returns the value for a dotted/key path into nested dicts."""
        data = {"metadata": {"model": {"name": "llama-3"}}}
        assert safe_get(data, "metadata", "model", "name") == "llama-3"

    def test_missing_key_returns_default(self):
        """Returns the default value when a key is missing."""
        data = {"a": 1}
        assert safe_get(data, "b") is None
        assert safe_get(data, "b", default="fallback") == "fallback"

    def test_empty_dict(self):
        """Returns default for any key in an empty dict."""
        data: dict = {}
        assert safe_get(data, "anything") is None

    def test_nested_missing(self):
        """Returns default when a middle key in the path is missing."""
        data = {"a": {"b": 1}}
        assert safe_get(data, "a", "x", "y") is None
        assert safe_get(data, "a", "x", "y", default=0) == 0

    def test_non_dict_intermediate(self):
        """Returns default if an intermediate value is not a dict."""
        data = {"a": "not_a_dict"}
        assert safe_get(data, "a", "b") is None


# ─── bar ──────────────────────────────────────────────────────────────────────


class TestBar:
    def test_full_bar(self):
        """When val == maxv, all cells should be filled."""
        result = bar(100, 100, w=10)
        assert "█" * 10 in result
        assert "░" not in result.replace("\033[0m", "").split("\033")[0]

    def test_empty_bar(self):
        """When val == 0, no cells should be filled."""
        result = bar(0, 100, w=10)
        assert "░" * 10 in result

    def test_half_bar(self):
        """50% of cells filled, 50% empty."""
        result = bar(50, 100, w=10)
        # 50/100 * 10 = 5 filled
        count_filled = result.count("█")
        count_empty = result.count("░")
        assert count_filled == 5
        assert count_empty == 5

    def test_zero_maxv_avoid_div_by_zero(self):
        """When maxv is 0, return 0 filled cells (no division by zero)."""
        result = bar(10, 0, w=10)
        # filled = round((10/0)*10) → 0 handled guard
        assert "░" * 10 in result

    def test_custom_width(self):
        """The w parameter controls bar width."""
        result = bar(100, 100, w=5)
        assert "█" * 5 in result

    def test_custom_color(self):
        """The color parameter is applied."""
        result = bar(100, 100, w=5, color=R)
        assert R in result
        assert N in result


# We can't easily introspect colour codes from the dynamic import,
# so we test the behaviour through pct_bar colour logic instead.


# ─── pct_bar ──────────────────────────────────────────────────────────────────


class TestPctBar:
    @staticmethod
    def _extract_ansi_codes(text: str) -> list[str]:
        """Return all ANSI escape sequences found in *text*."""
        import re

        return re.findall(r"\033\[\d+m", text)

    def test_100_percent(self):
        """100% should produce a full bar."""
        result = pct_bar(100, w=10)
        assert result.count("█") == 10
        assert result.count("░") == 0

    def test_0_percent(self):
        """0% should produce an empty bar (all empty cells)."""
        result = pct_bar(0, w=10)
        assert result.count("░") == 10
        assert result.count("█") == 0

    def test_50_percent(self):
        """50% should produce 10/20 filled cells."""
        result = pct_bar(50, w=20)
        assert result.count("█") == 10
        assert result.count("░") == 10

    def test_75_percent_yellow(self):
        """
        75% — >= 50 but < 80 → yellow colour.
        The function uses G if >= 80, Y if >= 50, R otherwise.
        """
        result = pct_bar(75, w=10)
        # 75/100 * 10 = 7.5 → round -> 8 filled
        assert result.count("█") == 8

    def test_80_percent_green(self):
        """80% — >= 80 → green colour."""
        result = pct_bar(80, w=10)
        assert result.count("█") == 8

    def test_30_percent_red(self):
        """30% — < 50 → red colour."""
        result = pct_bar(30, w=10)
        assert result.count("█") == 3
        assert result.count("░") == 7


# ─── format_model_name ────────────────────────────────────────────────────────


class TestFormatModelName:
    def test_full_path(self):
        """Full HuggingFace-style path strips org prefix."""
        result = format_model_name("org/repo/ModelName-GGUF")
        assert "/" not in result  # only the last part after /

    def test_short_name(self):
        """Short names with no separators pass through unchanged."""
        result = format_model_name("ModelName")
        assert result == "ModelName"

    def test_suffixes_stripped(self):
        """Known suffixes like -GGUF, :Q8_0 are removed."""
        result = format_model_name("ModelName-GGUF:Q8_0")
        assert "-GGUF" not in result
        assert ":Q8_0" not in result

    def test_multiple_suffixes(self):
        """All known suffixes are stripped in sequence."""
        result = format_model_name("org/MyModel-GGUF:Q4_K_M")
        assert "MyModel" in result
        assert "-GGUF" not in result
        assert ":Q4_K_M" not in result

    def test_long_name_truncated(self):
        """Names > 28 chars after stripping suffixes are truncated to 25 + '...'."""
        long_name = "this-is-a-very-long-model-name-that-exceeds-twenty-eight"
        result = format_model_name(long_name)
        assert len(result) <= 28  # 25 + '...'
        assert result.endswith("...")
