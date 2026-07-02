#!/usr/bin/env python3
"""
Banco de pruebas avanzado para llama-server usando /v1/chat/completions.
Mide latencia total, TTFT (Time to First Token), velocidad de generación (TPOP)
y recopila métricas en escenarios secuenciales y concurrentes.
"""

import argparse
import concurrent.futures
from datetime import datetime
import json
import os
import re
import statistics
import sys
import time
import requests

# =============================================================================
# COLORES ANSI
# =============================================================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

NO_COLOR = os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb" or not sys.stdout.isatty()

def c(color: str, text: str) -> str:
    if NO_COLOR:
        return text
    return f"{color}{text}{Color.RESET}"

# =============================================================================
# COMPATIBILIDAD DE ENDPOINTS
# =============================================================================
def get_root_url(url):
    if url.endswith("/v1"):
        return url[:-3]
    if url.endswith("/v1/"):
        return url[:-4]
    return url

# =============================================================================
# CAPTURA DE MÉTRICAS DEL SERVIDOR
# =============================================================================
def is_healthy(server_url):
    root_url = get_root_url(server_url)
    try:
        r = requests.get(f"{root_url}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except:
        return False

def get_props(server_url):
    root_url = get_root_url(server_url)
    try:
        r = requests.get(f"{root_url}/props", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_slots(server_url):
    root_url = get_root_url(server_url)
    try:
        r = requests.get(f"{root_url}/slots", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_metrics(server_url):
    root_url = get_root_url(server_url)
    try:
        r = requests.get(f"{root_url}/metrics", timeout=5)
        return r.text if r.status_code == 200 else None
    except:
        return None

def parse_metrics(text):
    if not text:
        return {}
    parsed = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            val = parts[1]
            if "{" in key:
                key = key.split("{")[0]
            # Normalizar claves de métricas (reemplazar ':' por '_') para soportar formato llamacpp:metric
            key = key.replace(":", "_")
            try:
                parsed[key] = float(val)
            except ValueError:
                parsed[key] = val
        except:
            continue
    return parsed

# =============================================================================
# EJECUCIÓN DE INFERENCIA
# =============================================================================
def send_chat_request(server_url, prompt, config):
    """Envía una solicitud de chat y mide TTFT, TPOP y latencia total."""
    messages = []
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": config["system_prompt"]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "n_predict": config["n_predict"],
        "stream": config["stream"],
    }
    
    if config["stream"]:
        payload["stream_options"] = {"include_usage": True}

    root_url = get_root_url(server_url)
    endpoint = f"{root_url}/v1/chat/completions"

    start_time = time.time()
    elapsed = 0.0
    ttft = None
    tokens_predicted = 0
    tokens_evaluated = 0
    stream_tokens_count = 0
    content = ""
    reasoning_content = ""
    success = False
    error_msg = ""

    try:
        if config["stream"]:
            r = requests.post(endpoint, json=payload, timeout=config["timeout"], stream=True)
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "usage" in chunk and chunk["usage"]:
                            tokens_predicted = chunk["usage"].get("completion_tokens", tokens_predicted)
                            tokens_evaluated = chunk["usage"].get("prompt_tokens", tokens_evaluated)
                        
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            delta_content = delta.get("content", "") or ""
                            reasoning_content_delta = delta.get("reasoning_content", "") or ""
                            
                            if delta_content or reasoning_content_delta:
                                if ttft is None:
                                    ttft = time.time() - start_time
                                
                                if delta_content:
                                    content += delta_content
                                if reasoning_content_delta:
                                    reasoning_content += reasoning_content_delta
                                
                                stream_tokens_count += 1
                    except:
                        pass
            
            # Si no obtuvimos tokens_predicted de la estructura usage, usamos la cuenta local
            if not tokens_predicted:
                tokens_predicted = stream_tokens_count
                
            elapsed = time.time() - start_time
            success = True
        else:
            r = requests.post(endpoint, json=payload, timeout=config["timeout"])
            elapsed = time.time() - start_time
            r.raise_for_status()
            data = r.json()
            message_dict = data.get("choices", [{}])[0].get("message", {})
            content = message_dict.get("content", "") or ""
            reasoning_content = message_dict.get("reasoning_content", "") or ""
            tokens_predicted = data.get("usage", {}).get("completion_tokens", 0)
            tokens_evaluated = data.get("usage", {}).get("prompt_tokens", 0)
            success = True

    except Exception as e:
        success = False
        error_msg = str(e)
        elapsed = time.time() - start_time

    # Combinamos el contenido de razonamiento con el de la respuesta final para el reporte de calidad
    if reasoning_content:
        content = f"<think>\n{reasoning_content}\n</think>\n{content}"

    # Cálculo de velocidad de generación (tokens/s sobre la fase de generación pura)
    tpop = 0.0
    if success and tokens_predicted > 0:
        if config["stream"] and ttft is not None:
            gen_time = elapsed - ttft
            tpop = (tokens_predicted - 1) / gen_time if gen_time > 0 else 0.0
        else:
            tpop = tokens_predicted / elapsed if elapsed > 0 else 0.0

    return {
        "success": success,
        "error": error_msg,
        "elapsed": elapsed,
        "ttft": ttft,
        "tpop": tpop,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
        "content": content,
        "prompt": prompt,
    }

# =============================================================================
# GESTIÓN DEL BENCHMARK
# =============================================================================
def run_benchmark(server_url, prompts, config):
    n_requests = config["n_requests"]
    concurrency = config["concurrency"]
    
    tasks_to_run = []
    for prompt in prompts:
        for idx in range(n_requests):
            tasks_to_run.append((prompt, idx + 1))
            
    total_tasks = len(tasks_to_run)
    print(c(Color.CYAN, f"\n🚀 Iniciando banco de pruebas con {len(prompts)} prompts..."))
    print(f"   Total solicitudes: {total_tasks} ({n_requests} ejecuciones por prompt)")
    print(f"   Concurrencia: {concurrency} thread(s)")
    print(f"   Tokens max a predecir: {config['n_predict']}")
    print(f"   Modo streaming: {config['stream']}")
    print(f"   Parámetros → Temp: {config['temperature']}, Top-p: {config['top_p']}, Top-k: {config['top_k']}\n")
    
    results = []
    wall_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {}
        for idx, (prompt, run_id) in enumerate(tasks_to_run):
            future = executor.submit(send_chat_request, server_url, prompt, config)
            future_to_task[future] = (prompt, run_id, idx + 1)
            
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            prompt, run_id, task_num = future_to_task[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "success": False,
                    "error": str(exc),
                    "elapsed": 0.0,
                    "ttft": None,
                    "tpop": 0.0,
                    "tokens_predicted": 0,
                    "tokens_evaluated": 0,
                    "content": "",
                    "prompt": prompt,
                }
            
            res["run_id"] = run_id
            res["task_num"] = task_num
            results.append(res)
            
            completed_count += 1
            status_char = c(Color.GREEN, "✅") if res["success"] else c(Color.RED, "❌")
            if res["success"]:
                speed_info = f"{res['tokens_predicted']} tok en {res['elapsed']:.2f}s"
                if config["stream"] and res["ttft"] is not None:
                    speed_info += f" (TTFT: {res['ttft']:.2f}s, TPOP: {res['tpop']:.1f} tok/s)"
                else:
                    speed_info += f" ({res['tokens_predicted']/res['elapsed']:.1f} tok/s)"
                print(f"[{completed_count}/{total_tasks}] {status_char} T{task_num} (Prompt {prompts.index(prompt)+1}, Ejec {run_id}): {speed_info}")
            else:
                print(f"[{completed_count}/{total_tasks}] {status_char} T{task_num} (Prompt {prompts.index(prompt)+1}, Ejec {run_id}): Error: {res['error']}")
                
    wall_duration = time.time() - wall_start
    return results, wall_duration

# =============================================================================
# REPORTE DE RESULTADOS
# =============================================================================
def show_results(results, wall_duration, props, slots, metrics_text, config):
    print("\n" + "="*90)
    print(c(Color.BOLD + Color.CYAN, "📊 INFORME DETALLADO DEL BANCO DE PRUEBAS"))
    print("="*90)

    if props:
        print(c(Color.BOLD, "\n🖥️  INFORMACIÓN DEL SERVIDOR:"))
        print(f"   Modelo:           {props.get('model_alias', 'N/A')}")
        print(f"   Build Info:       {props.get('build_info', 'N/A')}")
        print(f"   Contexto Max:     {props.get('default_generation_settings', {}).get('n_ctx', 'N/A')}")
        
    if slots:
        processing = sum(1 for s in slots if s.get('is_processing', False))
        print(f"   Slots totales:    {len(slots)}")
        print(f"   Slots en uso:     {processing}/{len(slots)}")

    metrics = parse_metrics(metrics_text)
    if metrics:
        print(c(Color.BOLD, "\n⚡ MÉTRICAS CLAVE (Prometheus):"))
        metric_keys = {
            "llamacpp_prompt_tokens_seconds": "Velocidad evaluación prompt (tok/s)",
            "llamacpp_predicted_tokens_seconds": "Velocidad generación (tok/s)",
            "llamacpp_kv_cache_usage_ratio": "Uso de KV Cache (ratio)",
            "llamacpp_requests_processing": "Solicitudes procesándose actualmente",
            "llamacpp_tokens_predicted_total": "Total tokens predecidos",
            "llamacpp_prompt_tokens_total": "Total tokens de prompt evaluados"
        }
        for key, desc in metric_keys.items():
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    print(f"   • {desc:<40}: {val:.4f}" if val < 1.0 else f"   • {desc:<40}: {val:.2f}")
                else:
                    print(f"   • {desc:<40}: {val}")

    good = [r for r in results if r["success"]]
    print(c(Color.BOLD, f"\n📈 ESTADÍSTICAS GENERALES DE INFERENCIA:"))
    print(f"   • Solicitudes exitosas: {len(good)}/{len(results)}")
    
    if len(results) > 0:
        success_rate = (len(good) / len(results)) * 100
        print(f"   • Tasa de éxito:        {success_rate:.1f}%")

    if good:
        latencies = [r["elapsed"] for r in good]
        tokens_pred = [r["tokens_predicted"] for r in good]
        tokens_eval = [r["tokens_evaluated"] for r in good]
        
        print(f"   • Latencia total (s)  → Media: {statistics.mean(latencies):.3f}s | Min: {min(latencies):.3f}s | Max: {max(latencies):.3f}s")
        if len(latencies) > 1:
            print(f"                           Desv. Estándar: {statistics.stdev(latencies):.3f}s")
            
        ttfts = [r["ttft"] for r in good if r["ttft"] is not None]
        if ttfts:
            print(f"   • TTFT (Latencia 1º token) → Media: {statistics.mean(ttfts):.3f}s | Min: {min(ttfts):.3f}s | Max: {max(ttfts):.3f}s")
            if len(ttfts) > 1:
                print(f"                                Desv. Estándar: {statistics.stdev(ttfts):.3f}s")
        
        tpops = [r["tpop"] for r in good if r["tpop"] > 0]
        if tpops:
            print(f"   • Velocidad Gen (tok/s)  → Media: {statistics.mean(tpops):.2f} t/s | Min: {min(tpops):.2f} t/s | Max: {max(tpops):.2f} t/s")
            if len(tpops) > 1:
                print(f"                                Desv. Estándar: {statistics.stdev(tpops):.2f} t/s")

        total_tokens_generated = sum(tokens_pred)
        total_tokens_evaluated = sum(tokens_eval)
        print(f"   • Tokens totales         → Prompts: {total_tokens_evaluated} | Generados: {total_tokens_generated}")
        
        if config["concurrency"] > 1:
            agg_throughput = total_tokens_generated / wall_duration if wall_duration > 0 else 0
            print(f"   • Rendimiento Concurrente→ Tiempo total: {wall_duration:.2f}s | Throughput Agregado: {agg_throughput:.2f} tok/s")

        # Tabla resumida de ejecuciones (corregido problema de alineación por colores ANSI)
        print("\n" + "-"*90)
        print(c(Color.BOLD, "📋 TABLA DE EJECUCIONES:"))
        print("-"*90)
        header = f"{'ID':<4} | {'Estado':<6} | {'Prompt (resumen)':<30} | {'Tokens':<10} | {'TTFT':<7} | {'TPOP':<9} | {'Latencia':<8}"
        print(c(Color.BOLD, header))
        print("-"*90)
        for r in results:
            status_raw = "OK" if r["success"] else "ERROR"
            status_str = c(Color.GREEN, f"{status_raw:<6}") if r["success"] else c(Color.RED, f"{status_raw:<6}")
            
            prompt_summary = r["prompt"][:27] + "..." if len(r["prompt"]) > 30 else r["prompt"].ljust(30)
            tokens_str = f"{r['tokens_evaluated']}/{r['tokens_predicted']}"
            ttft_str = f"{r['ttft']:.3f}s" if r["ttft"] is not None else "N/A"
            tpop_str = f"{r['tpop']:.1f}/s" if r["success"] and r["tpop"] > 0 else "N/A"
            lat_str = f"{r['elapsed']:.2f}s"
            
            row = f"T{r['task_num']:<3} | {status_str} | {prompt_summary:<30} | {tokens_str:<10} | {ttft_str:<7} | {tpop_str:<9} | {lat_str:<8}"
            print(row)
        print("-"*90)

        # Muestras de calidad
        print(c(Color.BOLD, "\n📝 MUESTRAS DE CALIDAD DE GENERACIÓN (primeras 3):"))
        for idx, r in enumerate(good[:3], 1):
            print(f"\n   --- Muestra {idx} (T{r['task_num']}) ---")
            print(f"   Prompt: {r['prompt']}")
            content = r['content'].strip()
            if not content:
                print(f"   Respuesta: {c(Color.YELLOW, '[Respuesta vacía]')}")
            else:
                # Detectar si contiene bloque de razonamiento (pensamiento)
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    response = content[think_match.end():].strip()
                    
                    # Mostrar resumen del pensamiento
                    think_preview = thinking.replace('\n', ' ')
                    think_preview = think_preview[:100] + "..." if len(think_preview) > 100 else think_preview
                    print(f"   Pensamiento: {c(Color.GRAY, f'[{len(thinking)} caracteres]')} \"{c(Color.GRAY, think_preview)}\"")
                    
                    if not response:
                        print(f"   Respuesta: {c(Color.YELLOW, '[Respuesta vacía tras el pensamiento]')}")
                        if r['tokens_predicted'] >= config['n_predict'] * 0.9:
                            print(f"            {c(Color.YELLOW, '⚠️  El modelo agotó los tokens disponibles (' + str(config['n_predict']) + ') durante el razonamiento.')}")
                            print(f"            {c(Color.YELLOW, '   Aumenta el límite con -n/--n-predict (ej: -n 1024).')}")
                    else:
                        if len(response) > 300:
                            response_disp = f"{response[:300]}... {c(Color.GRAY, f'(truncado, total {len(response)} caracteres)')}"
                        else:
                            response_disp = response
                        print(f"   Respuesta: {response_disp}")
                elif "<think>" in content:
                    # El tag de think está abierto pero no cerrado (truncado/incompleto)
                    parts = content.split("<think>", 1)
                    thinking = parts[1].strip()
                    think_preview = thinking.replace('\n', ' ')
                    think_preview = think_preview[:150] + "..." if len(think_preview) > 150 else think_preview
                    print(f"   Pensamiento (incompleto): {c(Color.GRAY, think_preview)}")
                    print(f"   {c(Color.YELLOW, '⚠️ ADVERTENCIA: El modelo se quedó sin tokens durante la fase de razonamiento.')}")
                    print(f"                  Intenta aumentar el límite de generación con -p/--n-predict.")
                    print(f"   Respuesta: {c(Color.RED, '[No se generó respuesta debido a la interrupción]')}")
                else:
                    # Respuesta normal sin razonamiento
                    if len(content) > 300:
                        content_disp = f"{content[:300]}... {c(Color.GRAY, f'(truncado, total {len(content)} caracteres)')}"
                    else:
                        content_disp = content
                    print(f"   Respuesta: {content_disp}")

    print("\n" + "="*90)
    print(c(Color.BOLD + Color.GREEN, "✅ Prueba completada."))
    print("="*90)

# =============================================================================
# EXPORTACIÓN JSON
# =============================================================================
def save_results_to_json(results, wall_duration, props, slots, metrics_text, config, filepath):
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "server_url": config["server_url"],
            "n_requests": config["n_requests"],
            "n_predict": config["n_predict"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "system_prompt": config["system_prompt"],
            "stream": config["stream"],
            "concurrency": config["concurrency"],
            "timeout": config["timeout"]
        },
        "server": {
            "props": props,
            "slots": slots,
            "metrics": parse_metrics(metrics_text)
        },
        "statistics": {},
        "results": results
    }
    
    good = [r for r in results if r["success"]]
    if good:
        latencies = [r["elapsed"] for r in good]
        tokens_pred = [r["tokens_predicted"] for r in good]
        tokens_eval = [r["tokens_evaluated"] for r in good]
        ttfts = [r["ttft"] for r in good if r["ttft"] is not None]
        tpops = [r["tpop"] for r in good if r["tpop"] > 0]
        
        data["statistics"] = {
            "total_requests": len(results),
            "successful_requests": len(good),
            "success_rate": (len(good) / len(results)) * 100,
            "wall_duration_seconds": wall_duration,
            "latency": {
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "stddev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            },
            "tokens": {
                "prompt_total": sum(tokens_eval),
                "predicted_total": sum(tokens_pred)
            }
        }
        
        if ttfts:
            data["statistics"]["ttft"] = {
                "mean": statistics.mean(ttfts),
                "min": min(ttfts),
                "max": max(ttfts),
                "stddev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
            }
        if tpops:
            data["statistics"]["tpop"] = {
                "mean": statistics.mean(tpops),
                "min": min(tpops),
                "max": max(tpops),
                "stddev": statistics.stdev(tpops) if len(tpops) > 1 else 0.0
            }
        if config["concurrency"] > 1:
            data["statistics"]["aggregate_throughput"] = sum(tokens_pred) / wall_duration
            
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en JSON: {filepath}")
    except Exception as e:
        print(f"\n❌ Error al guardar resultados en JSON: {e}")

# =============================================================================
# CARGA DE PROMPTS
# =============================================================================
def load_prompts(prompts_arg):
    default_prompts = [
        "Explica qué es la inteligencia artificial en una frase.",
        "Escribe un breve poema sobre el otoño.",
        "¿Cuál es la capital de Francia y por qué es importante?",
        "Resume la trama de 'Cien años de soledad' en una oración.",
        "Dame tres ideas para empezar un negocio sostenible.",
    ]
    
    if not prompts_arg:
        return default_prompts
        
    if os.path.isfile(prompts_arg):
        try:
            with open(prompts_arg, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if prompts:
                return prompts
            else:
                print(c(Color.YELLOW, f"⚠️ El archivo de prompts '{prompts_arg}' está vacío. Usando prompts por defecto."))
                return default_prompts
        except Exception as e:
            print(c(Color.RED, f"❌ Error leyendo archivo de prompts '{prompts_arg}': {e}. Usando prompts por defecto."))
            return default_prompts
            
    prompts = [p.strip() for p in prompts_arg.split(",") if p.strip()]
    if prompts:
        return prompts
    return default_prompts

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Banco de pruebas completo para llama-server usando /v1/chat/completions."
    )
    parser.add_argument(
        "-u", "--url",
        default=os.getenv("LLAMA_HOST", "http://localhost:8080"),
        help="URL base del servidor llama.cpp (default: LLAMA_HOST o http://localhost:8080)"
    )
    parser.add_argument(
        "-r", "--n-requests",
        type=int,
        default=2,
        help="Número de ejecuciones por cada prompt (default: 2)"
    )
    parser.add_argument(
        "-n", "-p", "--n-predict",
        dest="n_predict",
        type=int,
        default=80,
        help="Máximo de tokens a predecir por solicitud (default: 80)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Temperatura de muestreo (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p para muestreo (default: 0.95)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k para muestreo (default: 40)"
    )
    parser.add_argument(
        "-s", "--system-prompt",
        default="Eres un asistente útil, conciso y claro.",
        help="System prompt opcional (default: 'Eres un asistente útil, conciso y claro.')"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Desactiva el modo de streaming (streaming permite medir TTFT y TPOP)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=1,
        help="Número de solicitudes simultáneas (default: 1)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Ruta al archivo JSON donde se guardarán los resultados"
    )
    parser.add_argument(
        "--prompts",
        help="Lista de prompts separados por comas o ruta a un archivo de texto con un prompt por línea"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Tiempo de espera en segundos para las solicitudes (default: 120)"
    )

    args = parser.parse_args()
    server_url = args.url.rstrip("/")

    print(f"🔍 Verificando conexión en {server_url}...")
    if not is_healthy(server_url):
        print(c(Color.RED, f"❌ No se puede conectar a {server_url}/health (o no es un endpoint de llama.cpp válido)"))
        sys.exit(1)
    print(c(Color.GREEN, "✅ Servidor OK."))

    prompts = load_prompts(args.prompts)

    config = {
        "server_url": server_url,
        "n_requests": args.n_requests,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "system_prompt": args.system_prompt,
        "stream": not args.no_stream,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "output": args.output
    }

    props = get_props(server_url)
    if props:
        print(c(Color.GREEN, f"✅ Modelo: {props.get('model_alias', 'N/A')}"))
    slots = get_slots(server_url)
    if slots:
        print(c(Color.GREEN, f"✅ Slots disponibles: {len(slots)}"))

    try:
        results, wall_duration = run_benchmark(server_url, prompts, config)
        results.sort(key=lambda x: x["task_num"])
        metrics_text = get_metrics(server_url)
        
        show_results(results, wall_duration, props, slots, metrics_text, config)
        
        if args.output:
            save_results_to_json(results, wall_duration, props, slots, metrics_text, config, args.output)

    except KeyboardInterrupt:
        print(c(Color.YELLOW, "\n⚠️ Prueba cancelada por el usuario (KeyboardInterrupt)."))
        sys.exit(130)

if __name__ == "__main__":
    main()