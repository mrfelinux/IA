#!/usr/bin/env python3
"""Comparador técnico de modelos locales para uso agéntico.

Analiza reportes JSON de diferentes modelos y genera informes comparativos.
Conecta a un modelo local (http://localhost:8080) para análisis inteligente.
"""

import json
import glob
import os
import sys
from typing import Any
from collections.abc import Callable

import requests
from datetime import datetime
from collections import OrderedDict

# ─── Type aliases (PEP 695 ─ Python 3.12+) ──────────────────────────────────

type ReportsDict = dict[str, Any]
type MetricsDict = dict[str, int | str]
type CategoryScores = dict[str, dict[str, float]]

# ─── ANSI ───────────────────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"; M = "\033[95m"
C = "\033[96m"; W = "\033[97m"; N = "\033[0m"; BO = "\033[1m"; DI = "\033[2m"
BL = "\033[90m"

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────
LOCAL_MODEL_URL = "http://localhost:8080"
LOCAL_MODEL_TIMEOUT = 120


def safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Obtiene un valor anidado de forma segura."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def bar(val: float, maxv: float, w: int = 20, color: str = G) -> str:
    """Barra horizontal."""
    filled = round((val / maxv) * w) if maxv else 0
    bar_str = "█" * filled + "░" * (w - filled)
    return f"{color}{bar_str}{N}"


def pct_bar(val: float, w: int = 20) -> str:
    """Barra de porcentaje 0-100."""
    return bar(val, 100, w, G if val >= 80 else (Y if val >= 50 else R))


def header(text: str) -> str:
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 70
    side = (cols - len(text) - 4) // 2
    return f"\n{BO}{C}{'═'*side}  {text}  {'═'*side}{N}"


def subheader(text: str) -> str:
    return f"\n{BO}{B}▸ {text}{N}"


def format_model_name(name: str) -> str:
    """Acorta el nombre del modelo para visualización."""
    short = name.split("/")[-1]
    for suffix in ["-GGUF", ":Q8_0", ":Q4_K_M", ":Q4_K_XL", ":UD-Q4_K_XL"]:
        short = short.replace(suffix, "")
    if len(short) > 28:
        short = short[:25] + "..."
    return short


# ─── CARGA DE REPORTES ──────────────────────────────────────────────────────
def load_reports(base_dir: str) -> ReportsDict:
    """Carga todos los reportes JSON del directorio."""
    reports: ReportsDict = {}
    json_files = sorted(glob.glob(os.path.join(base_dir, "reporte_*.json")))

    if not json_files:
        print(f"\n{Y}⚠ No se encontraron ficheros reporte_*.json en {base_dir}{N}")
        return reports

    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            m = metadata.get("nombre_modelo", "unknown")
            ts = metadata.get("fecha_ejecucion", "")
            key = f"{m} @ {ts}" if ts else m
            reports[key] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Y}⚠ Error cargando {os.path.basename(fpath)}: {e}{N}")

    return reports


# ─── ANÁLISIS CON MODELO LOCAL ──────────────────────────────────────────────
def query_local_model(prompt: str, max_tokens: int = 4096) -> str | None:
    """Consulta el modelo local para análisis inteligente."""
    try:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        response = requests.post(
            f"{LOCAL_MODEL_URL}/v1/completions",
            json=payload,
            timeout=LOCAL_MODEL_TIMEOUT
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("text", "")
        else:
            return None
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def generate_intelligent_analysis(reports_data: ReportsDict) -> str | None:
    """Genera análisis inteligente usando el modelo local."""
    # Preparar datos resumidos para el prompt
    summary = []
    for key, data in reports_data.items():
        model_name = safe_get(data, "metadata", "nombre_modelo", default="unknown")
        resumen = data.get("resumen", {})
        summary.append({
            "modelo": model_name,
            "exitosas": resumen.get("pruebas_exitosas", 0),
            "parciales": resumen.get("pruebas_parciales", 0),
            "fallidas": resumen.get("pruebas_fallidas", 0),
            "porcentaje": resumen.get("porcentaje_exito", "0%"),
            "tps": resumen.get("tps_promedio", 0),
            "tiempo": resumen.get("tiempo_total", 0)
        })

    # Obtener detalles de categorías débiles
    weak_categories = []
    for key, data in reports_data.items():
        model_name = safe_get(data, "metadata", "nombre_modelo", default="unknown")
        resultados = data.get("resultados_pruebas", {})
        for cat_id, cat_data in resultados.items():
            score = cat_data.get("valida_score", 0)
            if score < 0.7:
                weak_categories.append({
                    "modelo": model_name,
                    "categoria": cat_data.get("categoria", cat_id),
                    "score": score * 100,
                    "mensaje": cat_data.get("valida_msg", "")
                })

    prompt = f"""Eres un experto en análisis de modelos de lenguaje para uso agéntico.
Analiza los siguientes datos de rendimiento de modelos y genera un informe técnico detallado en español.

DATOS DE MODELOS:
{json.dumps(summary, indent=2, ensure_ascii=False)}

CATEGORÍAS CON BAJO RENDIMIENTO (score < 70%):
{json.dumps(weak_categories[:20], indent=2, ensure_ascii=False)}

Genera un informe con:
1. Resumen ejecutivo comparativo
2. Fortalezas y debilidades de cada modelo
3. Recomendaciones por caso de uso (coding, tool calling, razonamiento)
4. Análisis de consistencia y variabilidad
5. Recomendación final con justificación

Responde en formato markdown conciso."""

    print(f"\n{C}🤖 Consultando modelo local para análisis inteligente...{N}")
    analysis = query_local_model(prompt)

    if analysis:
        print(f"{G}✓ Análisis generado por modelo local{N}")
        return analysis
    else:
        print(f"{Y}⚠ Modelo local no disponible en {LOCAL_MODEL_URL}{N}")
        return None


def format_markdown_analysis(text: str | None) -> str | None:
    """Da formato ANSI atractivo a la salida markdown del análisis."""
    if not text:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            out.append(f"\n{BO}{C}{'─'*60}{N}")
            out.append(f"{BO}{C}  {stripped[2:].upper()}{N}")
            out.append(f"{C}{'─'*60}{N}")
        elif stripped.startswith("## "):
            out.append(f"\n{BO}{B}  ▸ {stripped[3:]}{N}")
        elif stripped.startswith("### "):
            out.append(f"\n{BO}{Y}    ▹ {stripped[4:]}{N}")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text_content = stripped[2:]
            bullet = "•"
            # Aplicar negritas inline **texto**
            parts = text_content.split("**")
            formatted = ""
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    formatted += f"{BO}{part}{N}"
                else:
                    formatted += part
            out.append(f"  {G}{bullet}{N} {formatted}")
        elif stripped.startswith("1.") or stripped.startswith("2.") or stripped.startswith("3.") or stripped.startswith("4.") or stripped.startswith("5."):
            out.append(f"\n{BO}{W}  {stripped}{N}")
        elif stripped == "":
            out.append("")
        else:
            parts = line.split("**")
            formatted = ""
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    formatted += f"{BO}{part}{N}"
                else:
                    formatted += part
            out.append(f"  {formatted}")
    return "\n".join(out)


# ─── SECCIONES DEL INFORME ──────────────────────────────────────────────────
def section_summary(reports: ReportsDict, groups: OrderedDict[str, list[ReportsDict]]) -> None:
    """Sección 1: Tabla resumen general."""
    print(header("1. RESUMEN GLOBAL POR MODELO"))
    # Compute dynamic column widths based on formatted model names
    model_names = [format_model_name(m) for m in groups.keys()]
    model_col_width = max((len(name) for name in model_names), default=0)
    # Cap width to avoid overly wide columns
    model_col_width = min(model_col_width, 40)
    # Define column widths
    col_model = model_col_width
    col_runs = 5
    col_success = 9
    col_partial = 9
    col_failure = 9
    col_pct = 8
    col_tps = 7
    # Build header line
    header_line = (f"{'MODELO':{col_model}} "
                   f"{'RUNS':>{col_runs}} "
                   f"{'EXITOSA':>{col_success}} "
                   f"{'PARCIAL':>{col_partial}} "
                   f"{'FALLIDA':>{col_failure}} "
                   f"{'%ÉXITO':>{col_pct}} "
                   f"{'TPS':>{col_tps}}")
    print(f"{BO}{W}  {header_line}{N}")
    # Separator line (matches header width + 2 leading spaces)
    sep_len = len(header_line)
    print(f"{BL}  {'─'*sep_len}{N}")

    for mname, runs in groups.items():
        short = format_model_name(mname)
        total_ok = sum(safe_get(r, "resumen", "pruebas_exitosas", default=0) for r in runs)
        total_par = sum(safe_get(r, "resumen", "pruebas_parciales", default=0) for r in runs)
        total_fail = sum(safe_get(r, "resumen", "pruebas_fallidas", default=0) for r in runs)
        total_all = total_ok + total_par + total_fail
        pct = (total_ok / total_all * 100) if total_all else 0
        pct_color = G if pct >= 85 else (Y if pct >= 70 else R)
        avg_tps = sum([safe_get(r, "resumen", "tps_promedio", default=0) for r in runs]) / len(runs) if runs else 0
        # Build row line with same column widths (% included inside col_pct)
        row_line = (f"  {short:<{col_model}} "
                    f"{len(runs):>{col_runs}} "
                    f"{total_ok:>{col_success}} "
                    f"{total_par:>{col_partial}} "
                    f"{total_fail:>{col_failure}} "
                    f"{pct_color}{pct:>{col_pct - 1}.1f}%{N} "
                    f"{avg_tps:>{col_tps}.1f}")
        print(row_line)


def section_metrics(groups: OrderedDict[str, list[ReportsDict]]) -> dict[str, dict[str, float]]:
    """Sección 2: Métricas de rendimiento con barras."""
    print(header("2. MÉTRICAS DE RENDIMIENTO (PROMEDIO POR MODELO)"))

    metrics = {}
    for mname, runs in groups.items():
        short_name = format_model_name(mname)
        tps_list = [safe_get(x, "resumen", "tps_promedio", default=0) for x in runs]
        time_list = [safe_get(x, "resumen", "tiempo_total", default=0) for x in runs]
        tokens_list = [safe_get(x, "resumen", "tokens_totales", default=0) for x in runs]
        pct_list = [safe_get(x, "resumen", "porcentaje_exito", default="0%") for x in runs]
        pct_num_list = []
        for p in pct_list:
            if isinstance(p, str):
                try:
                    pct_num_list.append(float(p.replace("%", "")))
                except ValueError:
                    pct_num_list.append(0)
            else:
                pct_num_list.append(float(p) if p else 0)

        metrics[short_name] = {
            "tps": sum(tps_list) / len(tps_list) if tps_list else 0,
            "time": sum(time_list) / len(time_list) if time_list else 0,
            "tokens": sum(tokens_list) / len(tokens_list) if tokens_list else 0,
            "pct": sum(pct_num_list) / len(pct_num_list) if pct_num_list else 0,
        }

    if not metrics:
        print(f"\n  {Y}No hay datos para mostrar{N}")
        return metrics

    max_tps = max(m["tps"] for m in metrics.values()) or 1

    for name, m in metrics.items():
        b = bar(m["tps"], max_tps, 25, C)
        pb = pct_bar(m["pct"], 25)
        print(f"\n{BO}{name}{N}")
        print(f"  {B}TPS{N}     {b} {C}{m['tps']:.1f}{N} tok/s")
        print(f"  {G}Éxito{N}  {pb} {G}{m['pct']:.1f}%{N}")
        print(f"  {Y}Tiempo{N}           {Y}{m['time']:.0f}{N} s total  |  {DI}{m['tokens']:.0f} tokens{N}")

    return metrics


def section_categories(groups: OrderedDict[str, list[ReportsDict]]) -> CategoryScores:
    """Sección 3: Desglose por categoría."""
    print(header("3. DESGLOSE POR CATEGORÍA (SCORE PROMEDIO)"))

    # Detectar categorías automáticamente de todos los reportes
    all_cats = OrderedDict()
    for mname, runs in groups.items():
        for run in runs:
            resultados = run.get("resultados_pruebas", {})
            for cat_id, cat_data in resultados.items():
                if cat_id not in all_cats and cat_data.get("categoria"):
                    all_cats[cat_id] = cat_data["categoria"]

    if not all_cats:
        print(f"\n  {Y}No se encontraron categorías{N}")
        return {}

    cat_scores = {}
    for mname, runs in groups.items():
        short = format_model_name(mname)
        cat_scores[short] = {}
        for cid, clabel in all_cats.items():
            scores = []
            for run in runs:
                pr = run.get("resultados_pruebas", {}).get(cid)
                if pr:
                    score = pr.get("valida_score", 0)
                    if score is not None:
                        scores.append(float(score))
            cat_scores[short][cid] = sum(scores) / len(scores) if scores else 0

    model_names = list(cat_scores.keys())
    cid_list = list(all_cats.keys())

    hdr = f"  {'CATEGORÍA':<22}"
    for mn in model_names:
        hdr += f"  {mn:>14}"
    print(f"\n{BO}{W}{hdr}{N}")
    print(f"  {BL}{'─'*22}{'─'*(16*len(model_names))}{N}")

    for cid in cid_list:
        line = f"  {all_cats[cid]:<22}"
        for mn in model_names:
            sc = cat_scores[mn].get(cid, 0) * 100
            c = G if sc >= 90 else (Y if sc >= 50 else R)
            line += f"  {c}{sc:>13.0f}%{N}"
        print(line)

    return cat_scores


def section_weaknesses(cat_scores: CategoryScores) -> None:
    """Sección 4: Mapa de debilidades."""
    print(header("4. MAPA DE DEBILIDADES (SCORE < 70%)"))

    any_weak = False
    all_cats = {}
    for mn, scores in cat_scores.items():
        for cid, sc in scores.items():
            if cid not in all_cats:
                all_cats[cid] = cid

    for cid in all_cats:
        weak_models = []
        for mn in cat_scores:
            sc = cat_scores[mn].get(cid, 0) * 100
            if sc < 70:
                weak_models.append((mn, sc))
        if weak_models:
            any_weak = True
            print(f"\n  {Y}{cid}{N}")
            for mn, sc in weak_models:
                b = bar(sc, 100, 15, Y if sc >= 40 else R)
                print(f"    {mn:<30} {b} {sc:.0f}%")

    if not any_weak:
        print(f"\n  {G}✓ Ninguna debilidad crítica (score < 70%) en los promedios{N}")


def section_variability(reports: ReportsDict, groups: OrderedDict[str, list[ReportsDict]]) -> None:
    """Sección 5: Análisis de variabilidad para modelos con múltiples runs."""
    print(header("5. ANÁLISIS DE VARIABILIDAD"))

    multi_run_models = {m: runs for m, runs in groups.items() if len(runs) >= 2}

    if not multi_run_models:
        print(f"\n  {Y}No hay modelos con múltiples ejecuciones para analizar variabilidad{N}")
        return

    for mname, runs in multi_run_models.items():
        short = format_model_name(mname)
        print(f"\n{BO}{short}{N} ({len(runs)} ejecuciones)")

        cat_var = {}
        for run in runs:
            resultados = run.get("resultados_pruebas", {})
            for cid, cat_data in resultados.items():
                score = cat_data.get("valida_score", 0)
                if score is not None:
                    if cid not in cat_var:
                        cat_var[cid] = {"scores": [], "label": cat_data.get("categoria", cid)}
                    cat_var[cid]["scores"].append(float(score))

        print(f"  {BO}{'CATEGORÍA':<22} {'PROM':>5} {'MIN':>5} {'MAX':>5} {'RANGO':>6}{N}")
        print(f"  {BL}{'─'*48}{N}")

        for cid, info in sorted(cat_var.items(), key=lambda x: -(max(x[1]["scores"]) - min(x[1]["scores"]))):
            scores = info["scores"]
            avg = sum(scores) / len(scores) * 100
            lo = min(scores) * 100
            hi = max(scores) * 100
            spread = hi - lo
            spc = R if spread > 50 else (Y if spread > 20 else G)
            print(f"  {info['label']:<22} {avg:>4.0f}% {lo:>4.0f}% {hi:>4.0f}% {spc}{spread:>4.0f}pp{N}")


def section_ranking(metrics: dict[str, dict[str, float]]) -> None:
    """Sección 6: Ranking final y recomendaciones."""
    print(header("6. RANKING FINAL Y RECOMENDACIONES"))

    if not metrics:
        print(f"\n  {Y}No hay datos para generar ranking{N}")
        return

    sorted_models = sorted(metrics.items(), key=lambda x: (-x[1]["pct"], -x[1]["tps"]))

    print(f"\n{BO}{'═'*46}{N}")
    print(f"{BO}{'  # MODELO':<33} {'%ÉXITO':>6}  {'TPS':>4}{N}")
    print(f"{BO}{'═'*46}{N}")

    for i, (name, m) in enumerate(sorted_models, 1):
        pc = G if m["pct"] >= 85 else (Y if m["pct"] >= 70 else R)
        print(f"{BO}  {i:<2} {name:<28} {pc}{m['pct']:>5.1f}%{N}  {C}{m['tps']:>4.1f}{N}")
    print(f"{BO}{'═'*46}{N}")

    # Recomendaciones generales
    print(f"""
{BO}RECOMENDACIONES GENERALES:{N}

  {G}► Para agentes críticos (tool calling):{N}
     Busca el modelo con mayor % de éxito en categorías de tool calling.
     La consistencia entre ejecuciones es clave.

  {G}► Para coding agéntico:{N}
     Prioriza modelos con alto score en generación de código y testing.
     Los modelos MoE suelen ser más eficientes en velocidad.

  {G}► Para razonamiento y contexto largo:{N}
     Modelos con mayor contexto nativo son preferibles.
     Evalúa la calidad de explicaciones y análisis.

  {Y}⚠ PRECAUCIÓN:{N} Las cuantizaciones afectan significativamente la calidad.
     Compara siempre con la misma cuantización cuando sea posible.
""")


def section_visual(metrics: dict[str, dict[str, float]]) -> None:
    """Sección 7: Gráfico comparativo visual."""
    print(header("7. COMPARATIVA VISUAL (FIABILIDAD vs VELOCIDAD)"))

    if not metrics:
        print(f"\n  {Y}No hay datos para mostrar{N}")
        return

    max_tps = max(m["tps"] for m in metrics.values()) or 1

    for mn, m in metrics.items():
        p = m["pct"]
        t = m["tps"]
        pc = G if p >= 90 else (Y if p >= 75 else R)
        tc = C if t >= 70 else (M if t >= 50 else Y)

        h_fill = round(p / 100 * 8)
        h_bar = "█" * h_fill + "░" * (8 - h_fill)

        w_fill = round(t / max_tps * 15)
        w_bar = "█" * w_fill + "░" * (15 - w_fill)

        print(f"\n  {BO}{mn:<28}{N}")
        print(f"  Fiabilidad {pc}{p:>5.1f}%{N}  ┃{pc}{h_bar}{N}┃")
        print(f"  Velocidad  {tc}{t:>5.1f} TPS{N}  ┃{tc}{w_bar}{N}┃")

    print(f"\n  {DI}Leyenda: Fiabilidad ≥90% {G}verde{N}{DI}, ≥75% {Y}amarillo{N}{DI}; Velocidad ≥70 {C}cyan{N}{DI}, ≥50 {M}magenta{N}{DI}")


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{BO}{M}{'█'*70}{N}")
    print(f"{BO}{M}█{' '*(68)}█{N}")
    print(f"{BO}{M}█  COMPARADOR TÉCNICO DE MODELOS LOCALES PARA USO AGÉNTICO  █{N}")
    print(f"{BO}{M}█{' '*(68)}█{N}")
    print(f"{BO}{M}{'█'*70}{N}\n")
    print(f"{DI}Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}{N}")
    print(f"{DI}Directorio: {base_dir}{N}\n")

    # Cargar reportes
    reports = load_reports(base_dir)
    if not reports:
        print(f"{R}✗ No se encontraron reportes para analizar{N}")
        sys.exit(1)

    print(f"{G}✓ Cargados {len(reports)} reportes{N}")

    # Agrupar por modelo
    groups = OrderedDict()
    for key, data in reports.items():
        m = safe_get(data, "metadata", "nombre_modelo", default="unknown")
        groups.setdefault(m, []).append(data)

    print(f"{G}✓ {len(groups)} modelos detectados{N}")

    # Ejecutar secciones
    section_summary(reports, groups)
    metrics = section_metrics(groups)
    cat_scores = section_categories(groups)
    section_weaknesses(cat_scores)
    section_variability(reports, groups)
    section_ranking(metrics)
    section_visual(metrics)

    # Análisis inteligente con modelo local
    print(header("8. ANÁLISIS INTELIGENTE (MODELO LOCAL)"))
    analysis = generate_intelligent_analysis(reports)
    if analysis:
        print(f"\n{G}{'━'*60}{N}")
        print(format_markdown_analysis(analysis))
        print(f"{G}{'━'*60}{N}")
    else:
        print(f"\n{Y}No se pudo generar análisis inteligente.{N}")
        print(f"{DI}Asegúrate de que el modelo local esté disponible en {LOCAL_MODEL_URL}{N}")

    # Footer
    print(f"\n{BO}{M}{'█'*70}{N}")
    print(f"{BO}{M}█{' '*(68)}█{N}")
    print(f"{BO}{M}█  FIN DEL INFORME — {len(reports)} ficheros analizados, {len(groups)} modelos comparados  █{N}")
    print(f"{BO}{M}█{' '*(68)}█{N}")
    print(f"{BO}{M}{'█'*70}{N}\n")


if __name__ == "__main__":
    main()
