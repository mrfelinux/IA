#!/usr/bin/env python3
"""
Comparador de reportes Hermes-Agent.

Lee todos los archivos hermes-*.json del directorio actual (o de una ruta
dada) y genera un informe comparativo vertical — legible en consola incluso
con nombres de modelo largos.

Uso:
  python hermes_comparador.py
  python hermes_comparador.py --dir ./reportes
  python hermes_comparador.py --latest      # solo los 2 más recientes
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ─── Colores ANSI ─────────────────────────────────────────────────────────────

COLOR = os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb" and sys.stdout.isatty()

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GRAY = "\033[90m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{C_RESET}" if COLOR else text


# ─── Ancho de terminal ────────────────────────────────────────────────────────

def _term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


# ─── Modelo de datos ──────────────────────────────────────────────────────────

@dataclass
class EscenarioEntry:
    name: str
    passed: bool
    total_latency_s: float
    tool_call_count: int
    errors: list[str]
    summary: str


@dataclass
class ReporteEntry:
    filepath: str
    timestamp: str
    model: str
    escenarios_exitosos: int
    escenarios_totales: int
    latencia_total_s: float
    tool_calls_totales: int
    metricas_servidor: dict[str, float]
    escenarios: list[EscenarioEntry]
    porcentaje_exito: float


# ─── Parseo ───────────────────────────────────────────────────────────────────

def cargar_reporte(path: str) -> ReporteEntry | None:
    try:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"{_c(C_RED, '⚠')} Error leyendo {path}: {e}")
        return None

    resumen = raw.get("resumen", {})
    escenarios_raw = raw.get("escenarios", [])

    escenarios = [
        EscenarioEntry(
            name=e.get("name", "?"),
            passed=e.get("passed", False),
            total_latency_s=e.get("total_latency_s", 0.0),
            tool_call_count=e.get("tool_call_count", 0),
            errors=e.get("errors", []),
            summary=e.get("summary", ""),
        )
        for e in escenarios_raw
    ]

    total = resumen.get("escenarios_totales", 0)
    exitosos = resumen.get("escenarios_exitosos", 0)
    pct_str = resumen.get("porcentaje_exito", "0%").replace("%", "")
    try:
        pct = float(pct_str)
    except ValueError:
        pct = (exitosos / total * 100) if total else 0.0

    return ReporteEntry(
        filepath=os.path.basename(path),
        timestamp=raw.get("timestamp", "?"),
        model=raw.get("model", "?"),
        escenarios_exitosos=exitosos,
        escenarios_totales=total,
        latencia_total_s=resumen.get("latencia_total_s", 0.0),
        tool_calls_totales=resumen.get("tool_calls_totales", 0),
        metricas_servidor=raw.get("metricas_servidor", {}),
        escenarios=escenarios,
        porcentaje_exito=pct,
    )


# ─── Helpers de formato ───────────────────────────────────────────────────────

MaxW = int

def _calc_cols(reportes: list[ReporteEntry], tw: int) -> MaxW:
    """Calcula el ancho máximo para el nombre del modelo según terminal y datos."""
    longest = max(len(r.model) for r in reportes)
    return min(longest + 4, max(28, tw // 2))


def _fmt_name(model: str, idx: int, mw: MaxW) -> str:
    """Formatea '#1 deepseek-v4-…-free' con truncado al centro."""
    tag = f"#{idx}"
    full = f"{tag} {model}"
    if len(full) <= mw:
        return full.ljust(mw)
    available = mw - len(tag) - 3
    if available < 4:
        return f"{tag} {model[:max(1, available-1)]}…".ljust(mw)
    half = (available - 1) // 2
    return f"{tag} {model[:half]}…{model[-(half):]}".ljust(mw)


def _pct_bar(pct: float, width: int = 10) -> str:
    """Barrita visual tipo [████░░░░]."""
    filled = round(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return _c(C_GREEN, bar) if pct >= 70 else _c(C_YELLOW, bar) if pct >= 40 else _c(C_RED, bar)


def _badge(passed: bool) -> str:
    return _c(C_GREEN, "OK") if passed else _c(C_RED, "FAIL")


def _bullet(n: int) -> str:
    return _c(C_GRAY, f"{n}.")


# ─── Reporte comparativo (formato vertical) ───────────────────────────────────

def generar_reporte(reportes: list[ReporteEntry]):
    if not reportes:
        print(f"{_c(C_RED, 'No hay reportes para comparar.')}")
        return

    tw = _term_width()
    mw = _calc_cols(reportes, tw)  # ancho máximo para nombre de modelo

    # ── Encabezado ──
    rule = "─" * min(tw, 72)
    print(f"\n{_c(C_BOLD + C_CYAN, rule)}")
    print(f"{_c(C_BOLD + C_CYAN, '  📊 COMPARADOR DE REPORTES HERMES-AGENT')}")
    print(f"{_c(C_BOLD + C_CYAN, rule)}")
    print(f"\n  {len(reportes)} reporte(s) encontrado(s):")
    for r in reportes:
        ts = r.timestamp[:19] if r.timestamp else "?"
        print(f"    {_c(C_GRAY, r.filepath):<55} {ts}")

    # ── Resumen por modelo ──
    print(f"\n  {_c(C_BOLD, 'Resumen')}")
    pcts = [r.porcentaje_exito for r in reportes]
    lats = [r.latencia_total_s for r in reportes]
    tools = [r.tool_calls_totales for r in reportes]

    for idx, r in enumerate(reportes, 1):
        pct = r.porcentaje_exito
        bar = _pct_bar(pct)
        name = _fmt_name(r.model, idx, mw)

        best_pct = max(pcts)
        best_lat = min(lats)
        best_tools = max(tools)

        pct_s = _c(C_GREEN, f"{pct:.1f}%") if pct == best_pct else f"{pct:.1f}%"
        lat_s = _c(C_GREEN, f"{r.latencia_total_s:.1f}s") if r.latencia_total_s == best_lat else f"{r.latencia_total_s:.1f}s"
        tools_s = _c(C_GREEN, str(r.tool_calls_totales)) if r.tool_calls_totales == best_tools else str(r.tool_calls_totales)

        esc = f"{r.escenarios_exitosos}/{r.escenarios_totales}"
        print(f"    {name} {bar}  {esc} escenarios  {pct_s}  {lat_s}  {tools_s} tools")

    # ── Desglose por escenario ──
    names_set: set[str] = set()
    for r in reportes:
        for e in r.escenarios:
            names_set.add(e.name)
    all_names = sorted(names_set)

    if all_names:
        print(f"\n  {_c(C_BOLD, 'Desglose por escenario')}")
        hdr_model = "Modelo".ljust(mw)
        print(f"      {hdr_model}  Estado   Latencia    Tools")
        for name in all_names:
            entries_for_name: list[tuple[int, ReporteEntry, EscenarioEntry]] = []
            for idx, r in enumerate(reportes, 1):
                for e in r.escenarios:
                    if e.name == name:
                        entries_for_name.append((idx, r, e))
                        break

            if not entries_for_name:
                continue

            print(f"\n    {_c(C_BOLD, name)}")
            for idx, r, e in entries_for_name:
                name_col = _fmt_name(r.model, idx, mw)
                status = _badge(e.passed)
                lat = f"{e.total_latency_s:.2f}s"
                tools_w = f"{e.tool_call_count} tool{'s' if e.tool_call_count != 1 else ''}"
                errs = _c(C_RED, f"  ⚠ {e.errors[0]}") if e.errors else ""
                print(f"      {name_col} {status}  {lat:>8}  {tools_w}{errs}")

    # ── Mejor rendimiento ──
    best_pct_idx = pcts.index(max(pcts))
    best = reportes[best_pct_idx]
    detalles = (
        f"{best.porcentaje_exito:.0f}% éxito, "
        f"{best.latencia_total_s:.1f}s total, "
        f"{best.tool_calls_totales} tool calls"
    )
    print(f"\n  {_c(C_BOLD, '🏆 Mejor rendimiento:')} {_c(C_GREEN, best.model)}  ({detalles})")

    # ── Rankings ──
    print(f"\n  {_c(C_BOLD, 'Rankings')}")
    rankings = [
        ("Éxito más alto", lambda r: r.porcentaje_exito, "sufijo", "%"),
        ("Latencia más baja", lambda r: r.latencia_total_s, "invert", "s"),
        ("Más tool calls", lambda r: r.tool_calls_totales, "sufijo", ""),
    ]

    for rank_name, key_fn, sort_mode, unit in rankings:
        if sort_mode == "invert":
            ranked = sorted(reportes, key=key_fn)
        else:
            ranked = sorted(reportes, key=key_fn, reverse=True)

        parts: list[str] = []
        for pos, rr in enumerate(ranked, 1):
            val = key_fn(rr)
            label = f"{pos}. {rr.model}: {val:.1f}{unit}" if unit else f"{pos}. {rr.model}: {int(val)}"
            if pos == 1:
                label = _c(C_GREEN, label)
            parts.append(label)

        line = f"    {rank_name:<22}  {' → '.join(parts)}"
        tw_line = _term_width()
        if len(line) > tw_line:
            line = f"    {rank_name:<22}"
            for p in parts:
                line += f"\n      {'':>22}  {p}"
        print(line)

    # ── Métricas del servidor ──
    metric_keys = [
        ("llamacpp_prompt_tokens_seconds", "Prompt eval (tok/s)"),
        ("llamacpp_predicted_tokens_seconds", "Generación (tok/s)"),
        ("llamacpp_kv_cache_usage_ratio", "KV Cache uso (ratio)"),
        ("llamacpp_tokens_predicted_total", "Tokens generados"),
        ("llamacpp_prompt_tokens_total", "Tokens prompt evaluados"),
    ]
    has_metrics = any(r.metricas_servidor for r in reportes)

    if has_metrics:
        print(f"\n  {_c(C_BOLD, '⚡ Métricas del servidor')}")

        # column width per model: max (#idx name length, value width) + pad
        val_width = 14
        col_headers: list[str] = []
        col_widths: list[int] = []
        for idx, r in enumerate(reportes, 1):
            h = f"#{idx} {r.model}"
            col_headers.append(h)
            col_widths.append(max(len(h) + 2, val_width + 2))

        label_w = 28
        indent = 2
        total_w = indent + label_w + 2 + sum(col_widths)
        tw = _term_width()

        if total_w <= tw:
            # ── Transposed table: metrics as rows, models as columns ──
            header = " " * indent + " " * (label_w + 2)
            for i, h in enumerate(col_headers):
                header += f"{h:>{col_widths[i]}}"
            print(f"{_c(C_GRAY, header)}")

            for key, label in metric_keys:
                vals = [r.metricas_servidor.get(key) for r in reportes]
                if all(v is None for v in vals):
                    continue
                numeric_vals = [v for v in vals if v is not None]
                best_v = max(numeric_vals) if numeric_vals else 0

                row = "  " + f"{_c(C_BOLD, label):>{label_w+2}}"
                for idx, r in enumerate(reportes, 1):
                    v = vals[idx - 1]
                    if v is None:
                        row += f"{'—':>{col_widths[idx-1]}}"
                        continue
                    if isinstance(v, float) and v < 1:
                        val_str = f"{v:.4f}"
                    else:
                        val_str = f"{v:,.1f}"
                    if v == best_v:
                        val_str = _c(C_GREEN, val_str)
                        row += f"{val_str:>{col_widths[idx-1]}}"
                    else:
                        row += f"{val_str:>{col_widths[idx-1]}}"
                print(row)
        else:
            # ── Fallback: one line per model per metric ──
            for key, label in metric_keys:
                vals = [r.metricas_servidor.get(key) for r in reportes]
                if all(v is None for v in vals):
                    continue
                numeric_vals = [v for v in vals if v is not None]
                best_v = max(numeric_vals) if numeric_vals else 0

                print(f"\n    {_c(C_GRAY, label)}")
                for idx, r in enumerate(reportes, 1):
                    v = vals[idx - 1]
                    if v is None:
                        continue
                    name_col = _fmt_name(r.model, idx, mw)
                    if isinstance(v, float) and v < 1:
                        val_str = f"{v:.4f}"
                    else:
                        val_str = f"{v:,.1f}"
                    if v == best_v:
                        val_str = _c(C_GREEN, val_str)
                        print(f"      {name_col}  {val_str:>12}  ← mejor")
                    else:
                        print(f"      {name_col}  {val_str:>12}")

    print(f"\n{_c(C_GREEN, '✅ Comparación completada.')}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara múltiples reportes JSON de Hermes-Agent"
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directorio donde buscar reportes hermes-*.json (default: .)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Solo comparar los 2 reportes más recientes",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Uno o más archivos JSON específicos (opcional)",
    )

    args = parser.parse_args()

    if args.files:
        paths = args.files
    else:
        pattern = os.path.join(args.dir, "hermes-*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not paths:
        print(f"{_c(C_RED, '❌')} No se encontraron archivos hermes-*.json en {args.dir}")
        sys.exit(1)

    if args.latest:
        paths = paths[:2]

    reportes: list[ReporteEntry] = []
    for p in paths:
        r = cargar_reporte(p)
        if r:
            reportes.append(r)

    if not reportes:
        print(f"{_c(C_RED, '❌')} No se pudieron cargar reportes válidos")
        sys.exit(1)

    generar_reporte(reportes)


if __name__ == "__main__":
    main()
