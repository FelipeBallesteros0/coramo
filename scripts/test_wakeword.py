#!/home/felipe/coramo-env/bin/python3
"""
Diagnostico de wake word: muestra score en tiempo real mientras escuchas.
Usa esto para determinar que threshold funciona con tu voz real.
Ctrl+C para salir.
"""
import os
import sys
import subprocess
import numpy as np
from openwakeword.model import Model as WakeWordModel

OWW_MODEL_PATH  = os.path.expanduser("~/coramo/models/coramo.onnx")
SAMPLE_RATE     = 16000
CHUNK_SAMPLES   = 1280
AUDIO_DEVICE    = "default"

print(f"Cargando modelo {OWW_MODEL_PATH} ...")
oww = WakeWordModel(wakeword_models=[OWW_MODEL_PATH], inference_framework="onnx")
print("Modelo listo. Habla ahora — di 'coramo' varias veces.")
print("Formato: [score] barra visual  (threshold recomendado: donde llega tu voz)\n")

proc = subprocess.Popen([
    "arecord", "-q", "-D", AUDIO_DEVICE,
    "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
], stdout=subprocess.PIPE)

max_score = 0.0
try:
    while True:
        raw = proc.stdout.read(CHUNK_SAMPLES * 2)
        if len(raw) < CHUNK_SAMPLES * 2:
            break
        chunk  = np.frombuffer(raw, dtype=np.int16)
        score  = oww.predict(chunk).get("coramo", 0.0)
        if score > max_score:
            max_score = score

        # Solo mostrar si hay actividad
        if score > 0.05:
            bar = "█" * int(score * 40)
            marker = " *** DETECCION" if score >= 0.5 else (" >> cerca" if score >= 0.3 else "")
            print(f"  {score:.3f} |{bar:<40}|{marker}", flush=True)

except KeyboardInterrupt:
    print(f"\n--- Resumen ---")
    print(f"Score maximo observado: {max_score:.3f}")
    if max_score >= 0.5:
        print("✓ El threshold 0.5 funciona con tu voz")
    elif max_score >= 0.3:
        print(f"→ Baja OWW_THRESHOLD a {max_score*0.8:.2f} en coramo-assistant.py")
    elif max_score >= 0.1:
        print("→ El modelo detecta algo pero necesita reentrenamiento con tu voz real")
    else:
        print("→ El modelo no detecta tu voz — necesita reentrenamiento con grabaciones reales")
finally:
    proc.terminate()
    proc.wait()
