"""Latencia de STT/TTS de OpenAI y de LLM en nube (OpenAI y DeepSeek) con tool obligatoria. venv: ~/venvs/bench
Uso: source ~/.config/coramo/env && python bench_nube.py"""
import json, os, sys, time
from openai import OpenAI
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
TOOLS_OAI = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
oai = OpenAI()
ordenes = cargar_ordenes()

SOLO_LLM = os.environ.get("SOLO_LLM") == "1"

def stt(item):
    i, _, _ = item
    with open(DATOS / f"{i:02d}.wav", "rb") as f:
        return oai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f, language="es").text
if not SOLO_LLM:
    p50, p95, _ = medir(stt, ordenes); imprimir("STT nube gpt-4o-mini-transcribe", p50, p95)

def tts_primer_byte(item):
    _, texto, _ = item
    with oai.audio.speech.with_streaming_response.create(model="gpt-4o-mini-tts", voice="coral", input=texto, response_format="pcm") as r:
        for _chunk in r.iter_bytes(chunk_size=4096):
            return 1
if not SOLO_LLM:
    p50, p95, _ = medir(tts_primer_byte, ordenes); imprimir("TTS nube gpt-4o-mini-tts (primer byte)", p50, p95)

def llm_openai_compatible(nombre, cliente, modelo, variantes):
    """Chat completions con tool obligatoria; sirve para OpenAI y DeepSeek (API compatible).
    `variantes`: lista de dicts de parámetros extra a probar en orden hasta que uno funcione."""
    uso = {"in": 0, "out": 0}; elegida = {}
    def llamar(texto, extra):
        return cliente.chat.completions.create(model=modelo,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}],
                tools=TOOLS_OAI, **extra)
    for v in variantes:
        try:
            llamar(ordenes[0][1], v); elegida = v; break
        except Exception as ex:
            print(f"  {nombre} {modelo}: variante {v} rechazada: {str(ex)[:90]}")
    else:
        print(f"LLM nube {nombre} {modelo}: ninguna variante funcionó"); return
    def pedir(item):
        _, texto, _ = item
        r = llamar(texto, elegida)
        uso["in"] += r.usage.prompt_tokens; uso["out"] += r.usage.completion_tokens
        tc = r.choices[0].message.tool_calls or []
        return tc[0].function.name if tc else "(sin tool)"
    p50, p95, res = medir(pedir, ordenes)
    ac = sum(1 for (_, _, e), _, o in res if e == o)
    imprimir(f"LLM nube {nombre} {modelo}", p50, p95, f"| acierto {ac}/30 | tokens in {uso['in']} out {uso['out']} | params {elegida}")
    for (i, t, e), _, o in res:
        if e != o: print(f"  fallo {i:02d}: {t} -> esperado {e}, obtuvo {o}")

import os
V_OAI = [
    {"max_completion_tokens": 200, "tool_choice": "required", "reasoning_effort": "minimal"},
    {"max_completion_tokens": 200, "tool_choice": "required"},
    {"max_completion_tokens": 200, "tool_choice": "auto"},
    {"max_tokens": 200, "temperature": 0, "tool_choice": "required"},
]
for modelo in os.environ.get("OPENAI_CHAT_MODELS", "gpt-5.5").split(","):
    llm_openai_compatible("OpenAI", oai, modelo.strip(), V_OAI)
if os.environ.get("DEEPSEEK_API_KEY"):
    ds = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    V_DS = [
        {"max_tokens": 200, "temperature": 0, "tool_choice": "required", "extra_body": {"thinking": {"type": "disabled"}}},
        {"max_tokens": 200, "temperature": 0, "tool_choice": "auto", "extra_body": {"thinking": {"type": "disabled"}}},
        {"max_tokens": 200, "tool_choice": "auto"},
    ]
    for modelo in os.environ.get("DEEPSEEK_CHAT_MODELS", "deepseek-flash").split(","):
        llm_openai_compatible("DeepSeek", ds, modelo.strip(), V_DS)
