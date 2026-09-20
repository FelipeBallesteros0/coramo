"""Latencia y acierto de tool de Qwen3-8B en llama-server. venv: ~/venvs/bench"""
import json, sys, requests
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir
URL = "http://127.0.0.1:8080/v1/chat/completions"
TOOLS = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
def pedir(item):
    _, texto, _ = item
    r = requests.post(URL, json={"model": "x", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}],
                                 "tools": TOOLS, "tool_choice": "required", "temperature": 0, "max_tokens": 80}, timeout=60).json()
    tc = r["choices"][0]["message"].get("tool_calls") or []
    return tc[0]["function"]["name"] if tc else "(sin tool)"
p50, p95, res = medir(pedir, cargar_ordenes())
aciertos = sum(1 for (_, _, esperado), _, obtenido in res if esperado == obtenido)
imprimir("LLM local Qwen3-8B", p50, p95, f"| acierto {aciertos}/30")
for (i, t, e), dt, o in res:
    if e != o: print(f"  fallo {i:02d}: {t} -> esperado {e}, obtuvo {o}")
