# src/coramo_brain/coramo_brain/core/tools.py
"""Herramientas que el modelo de lenguaje puede elegir, y su traduccion a comandos.

Este modulo es la fuente unica de la verdad sobre que puede pedir el modelo.
No importa rclpy: se prueba solo.
"""
from __future__ import annotations
import json
from pathlib import Path


class ComandoInvalido(Exception):
    """La orden del modelo no se puede convertir en un comando seguro."""


GESTOS = ["abre", "cierra", "paz", "ok", "rock", "pulgar"]
POSES_BRAZO = ["reposo", "saludo", "extendido", "arriba", "abajo"]
MIRADAS = ["frente", "izquierda", "derecha", "arriba", "abajo"]

TOOLS = [
    {"type": "function", "function": {
        "name": "mano",
        "description": "Mueve la mano robotica: un gesto completo o dedos individuales en grados.",
        "parameters": {"type": "object", "properties": {
            "gesto": {"type": "string", "enum": GESTOS},
            "dedos": {"type": "object", "additionalProperties": {"type": "number"}}},
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "brazo",
        "description": "Mueve el brazo a una pose nombrada o a angulos articulares en grados.",
        "parameters": {"type": "object", "properties": {
            "pose": {"type": "string", "enum": POSES_BRAZO},
            "articulaciones": {"type": "object", "additionalProperties": {"type": "number"}}},
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "cabeza",
        "description": "Orienta la cabeza del robot.",
        "parameters": {"type": "object", "properties": {
            "mirar": {"type": "string", "enum": MIRADAS},
            "articulaciones": {"type": "object", "additionalProperties": {"type": "number"}}},
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "responder",
        "description": "Responde por voz cuando no hay accion fisica.",
        "parameters": {"type": "object", "properties": {"texto": {"type": "string"}},
                       "required": ["texto"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "detener",
        "description": "Parada inmediata de todos los motores.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def cargar_limites(ruta: str | Path) -> dict:
    """Lee joints.yaml sin depender de PyYAML: el formato es plano y conocido."""
    datos = {"articulaciones": {}, "gestos_mano": {}, "poses_cabeza": {}}
    seccion = None
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        sin_comentario = linea.split("#", 1)[0].rstrip()
        if not sin_comentario:
            continue
        if not sin_comentario.startswith(" "):
            seccion = sin_comentario.rstrip(":")
            continue
        clave, _, resto = sin_comentario.strip().partition(":")
        cuerpo = resto.strip().strip("{}")
        valores = {}
        for par in cuerpo.split(","):
            k, _, v = par.partition(":")
            if k.strip():
                valores[k.strip()] = float(v.strip())
        if seccion == "articulaciones":
            datos["articulaciones"][clave] = (valores["min"], valores["max"])
        elif seccion in ("gestos_mano", "poses_cabeza"):
            datos[seccion][clave] = valores
    return datos


def _validar_angulos(angulos: dict, limites: dict) -> None:
    for nombre, grados in angulos.items():
        if nombre not in limites["articulaciones"]:
            raise ComandoInvalido(f"articulacion desconocida: {nombre}")
        bajo, alto = limites["articulaciones"][nombre]
        if not bajo <= grados <= alto:
            raise ComandoInvalido(f"{nombre}={grados} fuera de [{bajo}, {alto}]")


def a_comando(nombre: str, args: dict, limites: dict) -> dict:
    """Convierte la eleccion del modelo en los campos de BodyCommand.

    Lanza ComandoInvalido si algo no encaja. Nunca recorta en silencio.
    """
    if nombre == "detener":
        return {"tool": "detener", "preset": "", "joint_names": [], "joint_positions_deg": []}

    if nombre == "mano":
        if "gesto" in args:
            gesto = args["gesto"]
            if gesto not in limites["gestos_mano"]:
                raise ComandoInvalido(f"gesto desconocido: {gesto}")
            angulos = limites["gestos_mano"][gesto]
            preset = gesto
        elif "dedos" in args:
            angulos, preset = dict(args["dedos"]), ""
        else:
            raise ComandoInvalido("mano sin gesto ni dedos")
    elif nombre == "cabeza":
        if "mirar" in args:
            mirada = args["mirar"]
            if mirada not in limites["poses_cabeza"]:
                raise ComandoInvalido(f"mirada desconocida: {mirada}")
            angulos, preset = limites["poses_cabeza"][mirada], mirada
        elif "articulaciones" in args:
            angulos, preset = dict(args["articulaciones"]), ""
        else:
            raise ComandoInvalido("cabeza sin mirar ni articulaciones")
    elif nombre == "brazo":
        if "pose" in args:
            if args["pose"] not in POSES_BRAZO:
                raise ComandoInvalido(f"pose desconocida: {args['pose']}")
            # Las poses del brazo las define el subproyecto C tras el inventario.
            return {"tool": "brazo", "preset": args["pose"], "joint_names": [], "joint_positions_deg": []}
        if "articulaciones" not in args:
            raise ComandoInvalido("brazo sin pose ni articulaciones")
        angulos, preset = dict(args["articulaciones"]), ""
    else:
        raise ComandoInvalido(f"herramienta desconocida: {nombre}")

    _validar_angulos(angulos, limites)
    nombres = sorted(angulos)
    return {"tool": nombre, "preset": preset,
            "joint_names": nombres,
            "joint_positions_deg": [float(angulos[n]) for n in nombres]}
