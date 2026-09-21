"""Medicion de aceptacion del subproyecto A.

Todo se correlaciona por `speech_end`, la marca del instante en que el usuario
dejo de hablar, que viaja dentro de los mensajes. Agrupar por orden de llegada
no sirve: el agente tarda en decidir y para entonces ya empezo el turno
siguiente. Emparejar por indice tampoco: si una orden no se transcribe, todo lo
demas queda corrido.

Uso: python3 medir_a3.py [segundos]
"""
import difflib
import re
import statistics
import sys
import unicodedata
from pathlib import Path

import rclpy
from rclpy.node import Node

from coramo_msgs.msg import BodyCommand, Event, Say, Transcript

ORDENES = Path.home() / "coramo" / "tools" / "bench" / "ordenes.txt"


def _seg(t) -> float:
    return round(t.sec + t.nanosec / 1e9, 3)


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9n ]+", " ", t).split())


class Medidor(Node):
    def __init__(self):
        super().__init__("medir_a3")
        self.tx = {}        # speech_end -> (recibido, texto)
        self.cmd = {}       # speech_end -> (recibido, tool)
        self.say = {}       # speech_end -> recibido
        self.primer_audio = []
        self.create_subscription(Transcript, "/speech/text", self._tx, 200)
        self.create_subscription(BodyCommand, "/body/command_safe", self._cmd, 200)
        self.create_subscription(Say, "/tts/say", self._say, 100)
        self.create_subscription(Event, "/coramo/event", self._ev, 300)

    def _tx(self, m):
        self.tx[_seg(m.speech_end)] = (_seg(m.header.stamp), m.text)

    def _cmd(self, m):
        self.cmd[_seg(m.speech_end)] = (_seg(m.header.stamp), m.tool)

    def _say(self, m):
        self.say[_seg(m.speech_end)] = _seg(m.header.stamp)

    def _ev(self, m):
        if m.name == "tts_first_audio":
            self.primer_audio.append(_seg(m.header.stamp))


def _fila(nombre, valores, meta):
    if not valores:
        print(f"{nombre:<28} {0:>3}        -        -  {meta:>5.2f}s  sin datos")
        return
    v = sorted(valores)
    p95 = v[int(round(0.95 * (len(v) - 1)))]
    print(f"{nombre:<28} {len(v):>3} {statistics.median(v):>7.2f}s {p95:>7.2f}s"
          f"  {meta:>5.2f}s  {'cumple' if p95 <= meta else 'NO cumple'}")


def main():
    segundos = float(sys.argv[1]) if len(sys.argv) > 1 else 210.0
    lineas = [l for l in ORDENES.read_text(encoding="utf-8").splitlines() if l.strip()]
    esperado = {_norm(l.rsplit("|", 1)[0]): l.rsplit("|", 1)[1].strip() for l in lineas}

    rclpy.init()
    m = Medidor()
    fin = m.get_clock().now().nanoseconds / 1e9 + segundos
    while rclpy.ok() and m.get_clock().now().nanoseconds / 1e9 < fin:
        rclpy.spin_once(m, timeout_sec=0.2)

    print(f"\ntranscripciones {len(m.tx)} de {len(lineas)} ordenes | "
          f"comandos {len(m.cmd)} | respuestas habladas {len(m.say)}\n")
    print(f"{'tramo':<28} {'n':>3} {'p50':>8} {'p95':>8}  {'meta':>6}  estado")
    _fila("cierre del turno", [r - s for s, (r, _t) in m.tx.items()], 0.80)
    _fila("TOTAL hasta la accion", [r - s for s, (r, _t) in m.cmd.items()], 1.50)
    _fila("hasta pedir la respuesta", [r - s for s, r in m.say.items()], 1.60)

    aciertos, fallos, sin_decision = 0, [], 0
    for s, (_r, texto) in sorted(m.tx.items()):
        clave = _norm(texto)
        cercana = difflib.get_close_matches(clave, esperado, n=1, cutoff=0.55)
        if not cercana:
            continue
        quiere = esperado[cercana[0]]
        if s in m.cmd:
            hizo = m.cmd[s][1]
        elif s in m.say:
            hizo = "responder"
        else:
            sin_decision += 1
            continue
        if hizo == quiere:
            aciertos += 1
        else:
            fallos.append((texto, quiere, hizo))

    n = aciertos + len(fallos)
    print(f"\nacierto de herramienta: {aciertos}/{n}"
          + (f"  ({100*aciertos/n:.0f} %)" if n else "")
          + (f" | sin decision: {sin_decision}" if sin_decision else ""))
    for texto, quiere, hizo in fallos:
        print(f"  «{texto}» esperaba {quiere}, eligio {hizo}")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
