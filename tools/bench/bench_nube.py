"""Latencia de STT/TTS de OpenAI y de Claude con tool obligatoria. venv: ~/venvs/bench
Uso: source ~/.config/coramo/env && python bench_nube.py"""
import json, sys, time
from openai import OpenAI
from anthropic import Anthropic
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
TOOLS_OAI = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
# mismas tools en formato Anthropic (input_schema en vez de parameters), estrictas
TOOLS_ANT = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "input_schema": {**t["function"]["parameters"], "required": t["function"]["parameters"].get("required", [])},
              "strict": True} for t in TOOLS_OAI]
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
oai, ant = OpenAI(), Anthropic()
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

def claude(modelo, extra):
    uso = {"in": 0, "out": 0, "cache": 0}
    def pedir(item):
        _, texto, _ = item
        r = ant.messages.create(model=modelo, max_tokens=200,
                                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                tools=TOOLS_ANT, tool_choice={"type": "any"},
                                messages=[{"role": "user", "content": texto}], **extra)
        uso["in"] += r.usage.input_tokens; uso["out"] += r.usage.output_tokens
        uso["cache"] += getattr(r.usage, "cache_read_input_tokens", 0) or 0
        for b in r.content:
            if b.type == "tool_use":
                return b.name
        return "(sin tool)"
    p50, p95, res = medir(pedir, ordenes)
    ac = sum(1 for (_, _, e), _, o in res if e == o)
    imprimir(f"LLM nube {modelo}", p50, p95, f"| acierto {ac}/30 | tokens in {uso['in']} (cache {uso['cache']}) out {uso['out']}")
    for (i, t, e), _, o in res:
        if e != o: print(f"  fallo {i:02d}: {t} -> esperado {e}, obtuvo {o}")

claude("claude-haiku-4-5", {})
try:
    claude("claude-sonnet-5", {"thinking": {"type": "disabled"}})
except Exception as ex:
    print("sonnet con thinking disabled rechazado:", str(ex)[:120]); claude("claude-sonnet-5", {"output_config": {"effort": "low"}})
