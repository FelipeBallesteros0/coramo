# servers/speech/app.py
"""Servidor de habla de CORAMO. Corre en ~/venvs/stt con Python 3.12.

Captura en continuo, detecta el final de cada turno con Silero y transcribe con
faster-whisper. Emite eventos por SSE. La marca t_speech_end es el instante en
que el usuario dejo de hablar: de ahi se miden todas las latencias.

Fuente de audio:
  CORAMO_FUENTE=mic            -> arecord desde el microfono (por defecto)
  CORAMO_FUENTE=/ruta/a/wavs   -> reproduce esos WAV en orden, para pruebas
"""
import asyncio
import json
import os
import queue
from collections import deque
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

FRECUENCIA = 16000
MUESTRAS_POR_TROZO = 512                 # 32 ms, lo que exige Silero
SILENCIO_FIN_S = float(os.environ.get("CORAMO_SILENCIO_S", "0.6"))
HABLA_MINIMA_S = 0.20
PREVIOS_TROZOS = 10            # 320 ms de audio previo al inicio del habla
TURNO_MAXIMO_S = 15.0
COMPUERTA_DBFS = float(os.environ.get("CORAMO_COMPUERTA_DBFS", "-45"))
FUENTE = os.environ.get("CORAMO_FUENTE", "mic")
REPETIR = int(os.environ.get("CORAMO_REPETIR", "1"))   # 0 = sin fin
DISPOSITIVO = os.environ.get("CORAMO_ALSA", "default")
SESIONES = Path.home() / "datos" / "sesiones"

app = FastAPI()
vad = load_silero_vad()
modelo = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

_eventos: queue.Queue = queue.Queue()
_turnos: queue.Queue = queue.Queue()
_silenciado = threading.Event()


def _emitir(**kw) -> None:
    _eventos.put(kw)


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    return 20.0 * np.log10(rms)


def _guardar(muestras: np.ndarray) -> str:
    ahora = datetime.now()
    carpeta = SESIONES / ahora.strftime("%Y-%m-%d")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / (ahora.strftime("%H-%M-%S-%f")[:-3] + ".wav")
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FRECUENCIA)
        w.writeframes((np.clip(muestras, -1, 1) * 32767).astype("<i2").tobytes())
    return str(ruta)


def _trozos_micro():
    p = subprocess.Popen(
        ["arecord", "-q", "-D", DISPOSITIVO, "-f", "S16_LE", "-r", str(FRECUENCIA),
         "-c", "1", "-t", "raw"], stdout=subprocess.PIPE)
    n = MUESTRAS_POR_TROZO * 2
    try:
        while True:
            crudo = p.stdout.read(n)
            if len(crudo) < n:
                break
            yield np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0
    finally:
        p.terminate()


_siguiente = [0.0]


def _a_ritmo(trozo: np.ndarray) -> np.ndarray:
    """Espera lo necesario para entregar el trozo a la misma velocidad que el
    microfono. Sin esto el reloj de audio adelanta al de pared y las latencias
    medidas salen sin sentido."""
    ahora = time.monotonic()
    if _siguiente[0] == 0.0:
        _siguiente[0] = ahora
    espera = _siguiente[0] - ahora
    if espera > 0:
        time.sleep(espera)
    _siguiente[0] += MUESTRAS_POR_TROZO / FRECUENCIA
    return trozo


def _trozos_archivos(carpeta: str):
    """Reproduce los WAV de la carpeta, separados por silencio.

    REPETIR=1 da una pasada, que es lo que se quiere para medir con un juego de
    ordenes exacto. REPETIR=0 repite sin fin, para desarrollar sin microfono.
    """
    vuelta = 0
    while REPETIR == 0 or vuelta < REPETIR:
        vuelta += 1
        for ruta in sorted(Path(carpeta).glob("*.wav")):
            with wave.open(str(ruta)) as w:
                datos = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            muestras = datos.astype(np.float32) / 32768.0
            for i in range(0, len(muestras) - MUESTRAS_POR_TROZO, MUESTRAS_POR_TROZO):
                yield _a_ritmo(muestras[i:i + MUESTRAS_POR_TROZO])
            for _ in range(int(FRECUENCIA * (SILENCIO_FIN_S + 0.5)) // MUESTRAS_POR_TROZO):
                yield _a_ritmo(np.zeros(MUESTRAS_POR_TROZO, dtype=np.float32))


def _transcribir() -> None:
    """Hilo trabajador: transcribe los turnos que cierra la captura.

    Va aparte para que la captura nunca se detenga; si lo hiciera, el reloj de
    audio se atrasaria respecto al de pared y las latencias medidas no valdrian.
    """
    while True:
        turno, t_fin = _turnos.get()
        segs, _info = modelo.transcribe(turno, language="es", beam_size=1,
                                        vad_filter=False)
        texto = " ".join(s.text.strip() for s in segs).strip()
        _emitir(type="transcript", text=texto, t_speech_end=t_fin,
                t_emitted=time.time(), confidence=1.0, wav=_guardar(turno))


def _bucle() -> None:
    trozos = _trozos_micro() if FUENTE == "mic" else _trozos_archivos(FUENTE)
    dentro = False
    buffer: list[np.ndarray] = []
    ultimo_habla = 0.0
    inicio = 0.0
    # Reloj de audio: avanza con las muestras, no con el reloj de pared. Con el
    # microfono coincide con el tiempo real porque arecord entrega a 16 kHz; con
    # archivos hace que la deteccion se comporte igual, y las pruebas sean
    # deterministas en vez de depender de lo rapido que vaya la maquina.
    t0 = time.time()
    muestras = 0
    # Cola con el audio inmediatamente anterior. Silero marca el inicio cuando
    # ya hay voz clara, asi que sin esto se pierde el ataque de la primera
    # palabra: "coramo cierra la mano" se transcribia "Decoramos Sierra La Mano".
    previos: deque = deque(maxlen=PREVIOS_TROZOS)
    for trozo in trozos:
        muestras += len(trozo)
        if _silenciado.is_set():
            dentro, buffer = False, []
            previos.clear()
            continue
        ahora = t0 + muestras / FRECUENCIA
        if not dentro:
            previos.append(trozo)
        if _dbfs(trozo) < COMPUERTA_DBFS and not dentro:
            continue
        prob = float(vad(torch.from_numpy(trozo), FRECUENCIA).item())
        hay_voz = prob > 0.5
        if hay_voz and not dentro:
            dentro = True
            buffer = list(previos)
            previos.clear()
            inicio = ahora - len(buffer) * MUESTRAS_POR_TROZO / FRECUENCIA
            ultimo_habla = ahora
            _emitir(type="speech_start", t=inicio)
        elif dentro:
            buffer.append(trozo)
            if hay_voz:
                ultimo_habla = ahora
            fin_por_silencio = ahora - ultimo_habla >= SILENCIO_FIN_S
            fin_por_limite = ahora - inicio >= TURNO_MAXIMO_S
            if fin_por_silencio or fin_por_limite:
                dentro = False
                t_fin = ultimo_habla
                _emitir(type="speech_end", t=t_fin)
                if ultimo_habla - inicio < HABLA_MINIMA_S:
                    buffer = []
                    continue
                turno = np.concatenate(buffer)
                buffer = []
                _turnos.put((turno, t_fin))


@app.on_event("startup")
def arrancar() -> None:
    threading.Thread(target=_transcribir, daemon=True).start()
    threading.Thread(target=_bucle, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "fuente": FUENTE, "silenciado": _silenciado.is_set(),
            "modelo": "large-v3-turbo"}


@app.post("/mute")
def mute():
    _silenciado.set()
    return {"silenciado": True}


@app.post("/unmute")
def unmute():
    _silenciado.clear()
    return {"silenciado": False}


@app.get("/events")
async def events():
    async def generar():
        while True:
            try:
                ev = _eventos.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(generar(), media_type="text/event-stream")
