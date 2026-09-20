"""Latencia de STT/TTS de OpenAI y de LLM en nube (OpenAI y DeepSeek) con tool obligatoria. venv: ~/venvs/bench
Uso: source ~/.config/coramo/env && python bench_nube.py"""
import json, sys, time
from openai import OpenAI
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
TOOLS_OAI = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
oai = OpenAI()
ordenes = cargar_ordenes()

def stt(item):
    i, _, _ = item
    with open(DATOS / f"{i:02d}.wav", "rb") as f:
        return oai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f, language="es").text
p50, p95, _ = medir(stt, ordenes); imprimir("STT nube gpt-4o-mini-transcribe", p50, p95)

def tts_primer_byte(item):
    _, texto, _ = item
    with oai.audio.speech.with_streaming_response.create(model="gpt-4o-mini-tts", voice="coral", input=texto, response_format="pcm") as r:
        for _chunk in r.iter_bytes(chunk_size=4096):
            return 1
p50, p95, _ = medir(tts_primer_byte, ordenes); imprimir("TTS nube gpt-4o-mini-tts (primer byte)", p50, p95)

def llm_openai_compatible(nombre, cliente, modelo, extra):
    """Chat completions con tool obligatoria; sirve para OpenAI y DeepSeek (API compatible)."""
    uso = {"in": 0, "out": 0}
    def pedir(item):
        _, texto, _ = item
        r = cliente.chat.completions.create(model=modelo, max_tokens=200, temperature=0,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}],
                tools=TOOLS_OAI, tool_choice="required", **extra)
        uso["in"] += r.usage.prompt_tokens; uso["out"] += r.usage.completion_tokens
        tc = r.choices[0].message.tool_calls or []
        return tc[0].function.name if tc else "(sin tool)"
    p50, p95, res = medir(pedir, ordenes)
    ac = sum(1 for (_, _, e), _, o in res if e == o)
    imprimir(f"LLM nube {nombre} {modelo}", p50, p95, f"| acierto {ac}/30 | tokens in {uso['in']} out {uso['out']}")
    for (i, t, e), _, o in res:
        if e != o: print(f"  fallo {i:02d}: {t} -> esperado {e}, obtuvo {o}")

import os
for modelo in os.environ.get("OPENAI_CHAT_MODELS", "gpt-5.5").split(","):
    try:
        llm_openai_compatible("OpenAI", oai, modelo.strip(), {"reasoning_effort": "minimal"})
    except Exception as ex:
        print(f"{modelo}: sin reasoning_effort ({str(ex)[:80]}); reintento sin el parámetro")
        llm_openai_compatible("OpenAI", oai, modelo.strip(), {})
if os.environ.get("DEEPSEEK_API_KEY"):
    ds = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    llm_openai_compatible("DeepSeek", ds, os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat"), {})
else:
    print("DeepSeek: sin DEEPSEEK_API_KEY, no se mide")
