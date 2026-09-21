# src/coramo_brain/coramo_brain/core/speech_client.py
"""Cliente del servidor de habla. Solo biblioteca estandar."""
from __future__ import annotations
import json
import urllib.request


class Troceador:
    """Arma eventos SSE completos a partir de trozos de red sueltos."""

    def __init__(self):
        self._resto = b""

    def alimentar(self, datos: bytes) -> list[dict]:
        self._resto += datos
        eventos = []
        while b"\n\n" in self._resto:
            bloque, self._resto = self._resto.split(b"\n\n", 1)
            for linea in bloque.split(b"\n"):
                if not linea.startswith(b"data:"):
                    continue
                try:
                    eventos.append(json.loads(linea[5:].strip()))
                except json.JSONDecodeError:
                    pass
        return eventos


def escuchar(url: str, al_evento, timeout_s: float = 65.0) -> None:
    """Se conecta al flujo y llama a al_evento(dict) por cada evento.

    Devuelve el control si la conexion se corta, para que quien llame reintente.
    """
    troceador = Troceador()
    with urllib.request.urlopen(f"{url.rstrip('/')}/events", timeout=timeout_s) as r:
        while True:
            trozo = r.read(1024)
            if not trozo:
                return
            for ev in troceador.alimentar(trozo):
                al_evento(ev)
