"""Latencia de Kokoro en GPU: tiempo hasta el primer audio por frase. venv: ~/venvs/tts"""
import sys
from kokoro import KPipeline
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir
VOZ = "ef_dora"  # voz femenina en español; alternativa em_alex
pipe = KPipeline(lang_code="e", device="cuda")
def primer_audio(item):
    _, texto, _ = item
    for _gs, _ps, audio in pipe(texto, voice=VOZ):
        return len(audio)  # el primer trozo ya se podría reproducir
p50, p95, _ = medir(primer_audio, cargar_ordenes())
imprimir("TTS local Kokoro " + VOZ, p50, p95)
