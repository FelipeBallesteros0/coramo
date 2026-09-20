"""Latencia y WER de faster-whisper large-v3-turbo en CUDA. venv: ~/venvs/stt"""
import os, sys, site
# rutas de cuBLAS/cuDNN instaladas por pip
for pkg in ("cublas", "cudnn"):
    for sp in site.getsitepackages():
        d = os.path.join(sp, "nvidia", pkg, "lib")
        if os.path.isdir(d):
            os.environ["LD_LIBRARY_PATH"] = d + ":" + os.environ.get("LD_LIBRARY_PATH", "")
from faster_whisper import WhisperModel
from jiwer import wer
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
modelo = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
def transcribir(item):
    i, _, _ = item
    segs, _info = modelo.transcribe(str(DATOS / f"{i:02d}.wav"), language="es", beam_size=1, vad_filter=False)
    return " ".join(s.text.strip() for s in segs)
p50, p95, res = medir(transcribir, cargar_ordenes())
ref = [t.lower() for _, t, _ in cargar_ordenes()]; hyp = [r.lower() for _, _, r in res]
imprimir("STT local whisper-turbo", p50, p95, f"| WER {wer(ref, hyp)*100:.1f} %")
for (i, t, _), _, h in res[:5]:
    print(f"  {i:02d} ref: {t} | hyp: {h}")
