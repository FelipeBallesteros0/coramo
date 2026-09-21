# src/coramo_brain/coramo_brain/core/safety.py
"""Filtro entre la decision del modelo y el cuerpo.

Rechaza, nunca recorta: un recorte silencioso convierte una orden mal entendida
en un movimiento inesperado cerca de una persona.
"""
from __future__ import annotations

SEPARACION_MINIMA_S = 0.10


class Filtro:
    def __init__(self, limites: dict, separacion_minima_s: float = SEPARACION_MINIMA_S):
        self._limites = limites
        self._separacion = separacion_minima_s
        self._ultimo = None
        self.detenido = False

    def parar(self) -> None:
        self.detenido = True

    def rearmar(self) -> None:
        self.detenido = False
        self._ultimo = None

    def revisar(self, cmd: dict, ahora: float) -> tuple[bool, str]:
        if cmd.get("tool") == "detener":
            self.parar()
            return True, ""
        if self.detenido:
            return False, "el robot esta detenido; hace falta rearmarlo"
        if self._ultimo is not None and ahora - self._ultimo < self._separacion:
            return False, "dos comandos demasiado seguidos"

        nombres = cmd.get("joint_names") or []
        grados = cmd.get("joint_positions_deg") or []
        if len(nombres) != len(grados):
            return False, "la lista de articulaciones y la de angulos no coinciden"
        for nombre, valor in zip(nombres, grados):
            if nombre not in self._limites["articulaciones"]:
                return False, f"articulacion desconocida: {nombre}"
            bajo, alto = self._limites["articulaciones"][nombre]
            if not bajo <= valor <= alto:
                return False, f"{nombre}={valor} fuera de [{bajo}, {alto}]"

        self._ultimo = ahora
        return True, ""
