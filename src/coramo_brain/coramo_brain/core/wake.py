# src/coramo_brain/coramo_brain/core/wake.py
"""Palabra de activacion y atajo de parada, sobre el texto ya transcrito.

Se hace por texto y no con un detector aparte porque en el hito 0 se midio que
whisper ya transcribe bien, y asi no hay un segundo modelo que mantener.
"""
from __future__ import annotations
import difflib
import re
import unicodedata
from dataclasses import dataclass

PALABRAS = ["coramo"]
PREFIJOS = ["hola", "hey", "oye", "ey"]
PARADAS = ["detente", "para", "parate", "alto", "alto ahi", "quieto", "no te muevas", "detenete"]
PARECIDO_MINIMO = 0.80


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9n ]+", " ", t).split())


@dataclass
class Resultado:
    activado: bool
    es_parada: bool
    orden: str


def _es_la_palabra(palabra: str) -> bool:
    for objetivo in PALABRAS:
        if palabra == objetivo:
            return True
        # Solo toleramos sufijos o letras cambiadas, no palabras mas cortas:
        # "romo" y "como" no deben activar al robot.
        if len(palabra) >= len(objetivo) and \
                difflib.SequenceMatcher(None, palabra, objetivo).ratio() >= PARECIDO_MINIMO:
            return True
    return False


def revisar(texto: str) -> Resultado:
    palabras = normalizar(texto).split()
    posicion = None
    for i, palabra in enumerate(palabras):
        if _es_la_palabra(palabra) and (i == 0 or palabras[i - 1] in PREFIJOS or i <= 2):
            posicion = i
            break
    if posicion is None:
        return Resultado(False, False, "")
    orden = " ".join(palabras[posicion + 1:]).strip()
    es_parada = any(orden == p or orden.startswith(p + " ") for p in PARADAS)
    return Resultado(True, es_parada, orden)
