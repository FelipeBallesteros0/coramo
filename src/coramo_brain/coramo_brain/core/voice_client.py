# src/coramo_brain/coramo_brain/core/voice_client.py
"""Cliente del servidor de voz. Solo biblioteca estandar: corre en Python 3.14."""
from __future__ import annotations
import json
import urllib.error
import urllib.request


class Voz:
    def __init__(self, url_voz: str, url_habla: str, timeout_s: float = 30.0):
        self._voz = url_voz.rstrip("/")
        self._habla = url_habla.rstrip("/")
        self._timeout = timeout_s

    def _post(self, url: str, datos: dict | None, timeout: float) -> dict:
        cuerpo = json.dumps(datos or {}).encode()
        req = urllib.request.Request(url, data=cuerpo, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def _silenciar(self, activar: bool) -> None:
        """Best effort: si el servidor de habla no esta, no impide hablar."""
        try:
            self._post(f"{self._habla}/{'mute' if activar else 'unmute'}", {}, 2.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

    def decir(self, texto: str) -> dict:
        self._silenciar(True)
        try:
            return self._post(f"{self._voz}/say", {"texto": texto}, self._timeout)
        finally:
            self._silenciar(False)
