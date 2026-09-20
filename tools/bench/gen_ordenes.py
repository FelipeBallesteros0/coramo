"""Genera ~/datos/ordenes/NN.wav (16 kHz mono) con Kokoro para cada orden. venv: ~/venvs/tts"""
import sys, numpy as np, soundfile as sf, torch
from kokoro import KPipeline
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, DATOS
DATOS.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="e", device="cuda")
for i, texto, _ in cargar_ordenes():
    voz = "ef_dora" if i % 2 else "em_alex"  # alterna dos voces
    partes = [np.asarray(a) for _, _, a in pipe(texto, voice=voz)]
    audio24 = np.concatenate(partes).astype(np.float32)
    audio16 = torch.nn.functional.interpolate(torch.tensor(audio24)[None, None], scale_factor=16000/24000, mode="linear")[0, 0].numpy()
    sf.write(DATOS / f"{i:02d}.wav", audio16, 16000, subtype="PCM_16")
print("generados", len(list(DATOS.glob("*.wav"))))
