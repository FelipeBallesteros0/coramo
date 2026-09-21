# servers/voice/app.py
"""Servidor de voz de CORAMO. Corre en ~/venvs/tts con Python 3.12.

Sintetiza por frases y reproduce por el parlante. Devuelve cuando sono el primer
audio, que es lo que percibe la persona, y cuando termino.
"""
import io
import subprocess
import time
import wave

import numpy as np
from fastapi import FastAPI
from kokoro import KPipeline
from pydantic import BaseModel

VOZ = "ef_dora"
FRECUENCIA = 24000
DISPOSITIVO = "default"

app = FastAPI()
pipe = KPipeline(lang_code="e", device="cuda")


class Peticion(BaseModel):
    texto: str
    voz: str = VOZ


def _reproducir(audio: np.ndarray) -> None:
    """Reproduce por el parlante. Lanza si no suena, en vez de callarlo.

    Bajo systemd hace falta XDG_RUNTIME_DIR para alcanzar PipeWire; sin el,
    aplay responde "Host is down" y el robot parece hablar sin que suene nada.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FRECUENCIA)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    r = subprocess.run(["aplay", "-q", "-D", DISPOSITIVO, "-"],
                       input=buf.getvalue(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"no se pudo reproducir: {r.stderr.decode().strip()[:120]}")


@app.on_event("startup")
def calentar() -> None:
    """Sintetiza una frase corta sin reproducirla.

    Sin esto la primera peticion real tarda 1,8 s en vez de 0,13 s, porque el
    modelo carga la voz la primera vez. Medido el 2026-09-20.
    """
    for _gs, _ps, _audio in pipe("listo", voice=VOZ):
        break


@app.get("/health")
def health():
    return {"ok": True, "voz": VOZ, "motor": "kokoro"}


@app.post("/say")
def say(p: Peticion):
    t0 = time.time()
    primero = None
    for _gs, _ps, audio in pipe(p.texto, voice=p.voz):
        if primero is None:
            primero = time.time()
        _reproducir(np.asarray(audio))
    if primero is None:
        return {"error": "sin audio", "t_first_audio": None, "t_done": time.time()}
    return {"t_first_audio": primero, "t_done": time.time(), "t_recibido": t0}
