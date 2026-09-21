# src/coramo_brain/coramo_brain/core/agent.py
"""Decide que herramienta corresponde a una orden. Solo biblioteca estandar."""
from __future__ import annotations
import json
import urllib.request
from dataclasses import dataclass

from coramo_brain.core import tools

SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en espanol y "
          "respondes SIEMPRE con exactamente una llamada a herramienta. Si la orden no "
          "mueve nada, usa responder. Nunca expliques tu razonamiento.")
NO_ENTENDI = "No entendi la orden"


class SinHerramienta(Exception):
    """La respuesta del modelo no trae una llamada a herramienta utilizable."""


def leer_respuesta(crudo: dict) -> tuple[str, dict]:
    """Saca (nombre, argumentos) de la respuesta del servidor. Nunca revienta por formato."""
    try:
        llamadas = crudo["choices"][0]["message"].get("tool_calls") or []
        if not llamadas:
            raise SinHerramienta("el modelo no llamo a ninguna herramienta")
        funcion = llamadas[0]["function"]
        return funcion["name"], json.loads(funcion.get("arguments") or "{}")
    except SinHerramienta:
        raise
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        raise SinHerramienta(f"respuesta ilegible: {e}") from e


class Backend:
    def elegir(self, texto: str) -> tuple[str, dict]:
        raise NotImplementedError


class Grabado(Backend):
    """Backend de pruebas: respuestas fijas, sin red ni GPU."""

    def __init__(self, respuestas: dict[str, tuple[str, dict]]):
        self._respuestas = respuestas

    def elegir(self, texto: str) -> tuple[str, dict]:
        if texto not in self._respuestas:
            raise SinHerramienta("sin respuesta grabada")
        return self._respuestas[texto]


class LlamaServer(Backend):
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout_s: float = 10.0):
        self._url = url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout_s

    def elegir(self, texto: str) -> tuple[str, dict]:
        cuerpo = json.dumps({
            "model": "coramo",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": texto}],
            "tools": tools.TOOLS, "tool_choice": "required",
            "temperature": 0, "max_tokens": 80,
        }).encode()
        req = urllib.request.Request(self._url, data=cuerpo, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return leer_respuesta(json.loads(r.read()))


@dataclass
class Decision:
    comando: dict | None
    texto: str
    herramienta: str


class Agente:
    def __init__(self, backend: Backend, limites: dict):
        self._backend = backend
        self._limites = limites

    def procesar(self, orden: str) -> Decision:
        try:
            nombre, args = self._backend.elegir(orden)
        except (SinHerramienta, OSError, TimeoutError) as e:
            return Decision(None, f"{NO_ENTENDI}.", "ninguna")
        if nombre == "responder":
            return Decision(None, str(args.get("texto", "")), "responder")
        try:
            return Decision(tools.a_comando(nombre, args, self._limites), "", nombre)
        except tools.ComandoInvalido as e:
            return Decision(None, f"No puedo hacer eso: {e}", nombre)
