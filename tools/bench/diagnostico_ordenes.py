"""Por que se pierden algunas de las 30 ordenes.

Corre la misma deteccion de habla y transcripcion del servidor, pero archivo por
archivo, para ver exactamente cual falla y en que paso. Se ejecuta en ~/venvs/stt.
"""
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

TROZO = 512
FRECUENCIA = 16000
SILENCIO_FIN_S = 0.6
HABLA_MINIMA_S = 0.20
PREVIOS = 10
COMPUERTA_DBFS = -45.0

ORDENES = Path.home() / "coramo" / "tools" / "bench" / "ordenes.txt"
AUDIOS = Path.home() / "datos" / "ordenes"

vad = load_silero_vad()
modelo = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9n ]+", " ", t).split())


def dbfs(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-12)


def turnos_de(ruta: Path):
    """Devuelve los turnos que el detector encontraria en este archivo."""
    with wave.open(str(ruta)) as w:
        datos = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    muestras = datos.astype(np.float32) / 32768.0
    # Igual que el servidor: el archivo y despues silencio para cerrar el turno.
    cola = np.zeros(int(FRECUENCIA * (SILENCIO_FIN_S + 0.5)), dtype=np.float32)
    muestras = np.concatenate([muestras, cola])

    dentro, buffer, previos, salida = False, [], [], []
    ultimo, inicio, reloj = 0.0, 0.0, 0.0
    for i in range(0, len(muestras) - TROZO, TROZO):
        trozo = muestras[i:i + TROZO]
        reloj += TROZO / FRECUENCIA
        if not dentro:
            previos.append(trozo)
            previos = previos[-PREVIOS:]
        if dbfs(trozo) < COMPUERTA_DBFS and not dentro:
            continue
        hay_voz = float(vad(torch.from_numpy(trozo.copy()), FRECUENCIA).item()) > 0.5
        if hay_voz and not dentro:
            dentro, buffer = True, list(previos)
            previos = []
            inicio, ultimo = reloj, reloj
        elif dentro:
            buffer.append(trozo)
            if hay_voz:
                ultimo = reloj
            if reloj - ultimo >= SILENCIO_FIN_S:
                dentro = False
                if ultimo - inicio >= HABLA_MINIMA_S:
                    salida.append(np.concatenate(buffer))
                buffer = []
    return salida


def main():
    lineas = [l for l in ORDENES.read_text(encoding="utf-8").splitlines() if l.strip()]
    problemas = []
    print(f"{'n':>3}  {'dur':>5}  {'nivel':>7}  turnos  transcripcion")
    for i, linea in enumerate(lineas, 1):
        esperado = linea.rsplit("|", 1)[0].strip()
        ruta = AUDIOS / f"{i:02d}.wav"
        with wave.open(str(ruta)) as w:
            dur = w.getnframes() / w.getframerate()
            datos = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        nivel = dbfs(datos.astype(np.float32) / 32768.0)
        trozos = turnos_de(ruta)
        if not trozos:
            print(f"{i:>3}  {dur:5.2f}  {nivel:6.1f}dB       0  (NINGUN TURNO)")
            problemas.append((i, esperado, "el detector no encuentra habla"))
            continue
        textos = []
        for t in trozos:
            segs, _ = modelo.transcribe(t, language="es", beam_size=1, vad_filter=False)
            textos.append(" ".join(s.text.strip() for s in segs).strip())
        marca = "" if len(trozos) == 1 else "  <-- PARTIDO EN VARIOS"
        print(f"{i:>3}  {dur:5.2f}  {nivel:6.1f}dB  {len(trozos):>6}  "
              f"«{' | '.join(textos)}»{marca}")
        if len(trozos) > 1:
            problemas.append((i, esperado, f"partido en {len(trozos)} turnos"))
        elif "coramo" not in norm(textos[0]):
            problemas.append((i, esperado, "la transcripcion pierde la palabra de activacion"))

    print(f"\nordenes problematicas: {len(problemas)} de {len(lineas)}")
    for i, esperado, motivo in problemas:
        print(f"  {i:02d}  «{esperado}»  ->  {motivo}")


if __name__ == "__main__":
    main()
