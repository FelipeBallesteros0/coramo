#!/home/felipe/coramo-env/bin/python3
"""
capture_hard_negatives.py — Captura clips de audio que confunden al modelo OWW.

Corre OWW continuamente. Cada vez que score > threshold, guarda el buffer de audio
en ~/hard_negatives/ como un WAV etiquetado como negativo.

Uso:
    NO hables durante la sesion. Todo lo capturado es un falso positivo.
    Deja correr 10-15 minutos con el ruido normal del cuarto.

    python scripts/capture_hard_negatives.py
"""

import os
import sys
import subprocess
import wave
import time
import collections
import numpy as np
from openwakeword.model import Model as WakeWordModel

OWW_MODEL_PATH  = os.path.expanduser("~/coramo/models/coramo.onnx")
OUTPUT_DIR      = os.path.expanduser("~/hard_negatives")
SAMPLE_RATE     = 16000
CHUNK_SAMPLES   = 1280        # 80ms por chunk (requerido por OWW)
BUFFER_CHUNKS   = 20          # ~1.6s de buffer circular
THRESHOLD       = 0.5         # score minimo para capturar
COOLDOWN_SECS   = 2.0         # segundos entre capturas para evitar duplicados

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Cargando modelo {OWW_MODEL_PATH} ...")
oww = WakeWordModel(wakeword_models=[OWW_MODEL_PATH], inference_framework="onnx")
print(f"Modelo listo. Guardando clips en {OUTPUT_DIR}")
print("NO hables. Deja correr 10-15 min con el ruido normal del cuarto.")
print("Ctrl+C para terminar.\n")

audio_buf   = collections.deque(maxlen=BUFFER_CHUNKS)
captured    = 0
last_save   = 0.0

proc = subprocess.Popen([
    "arecord", "-q", "-D", "default",
    "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
], stdout=subprocess.PIPE)

try:
    while True:
        raw = proc.stdout.read(CHUNK_SAMPLES * 2)
        if len(raw) < CHUNK_SAMPLES * 2:
            break

        chunk = np.frombuffer(raw, dtype=np.int16)
        audio_buf.append(chunk)
        score = oww.predict(chunk).get("coramo", 0.0)

        now = time.time()
        if score >= THRESHOLD and (now - last_save) >= COOLDOWN_SECS:
            last_save = now
            captured += 1
            fname = os.path.join(OUTPUT_DIR, f"neg_{int(now*1000)}.wav")
            samples = np.concatenate(list(audio_buf))
            with wave.open(fname, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(samples.tobytes())
            print(f"  [{captured:03d}] score={score:.3f} → {os.path.basename(fname)}", flush=True)

except KeyboardInterrupt:
    print(f"\n--- Fin ---")
    print(f"Clips capturados: {captured}")
    print(f"Directorio: {OUTPUT_DIR}")
finally:
    proc.terminate()
    proc.wait()
