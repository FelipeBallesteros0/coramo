# tools/latencias.py
"""Escucha /coramo/event y saca la tabla de latencias por turno.

Uso: python3 tools/latencias.py [segundos]   (por defecto 300)
"""
import statistics
import sys
import rclpy
from rclpy.node import Node
from coramo_msgs.msg import Event

TRAMOS = [("speech_end", "transcript", "cierre del turno"),
          ("transcript", "tool_chosen", "decision"),
          ("tool_chosen", "command_sent", "validacion y envio"),
          ("speech_end", "command_sent", "TOTAL hasta la accion"),
          ("speech_end", "tts_first_audio", "hasta oir la respuesta")]


class Medidor(Node):
    def __init__(self):
        super().__init__("latencias")
        self.turnos, self.actual = [], {}
        self.create_subscription(Event, "/coramo/event", self._al_llegar, 50)

    def _al_llegar(self, ev: Event) -> None:
        t = ev.header.stamp.sec + ev.header.stamp.nanosec / 1e9
        if ev.name == "speech_start" and self.actual:
            self.turnos.append(self.actual)
            self.actual = {}
        self.actual[ev.name] = t


def main():
    segundos = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    rclpy.init()
    m = Medidor()
    fin = m.get_clock().now().nanoseconds / 1e9 + segundos
    while rclpy.ok() and m.get_clock().now().nanoseconds / 1e9 < fin:
        rclpy.spin_once(m, timeout_sec=0.2)
    if m.actual:
        m.turnos.append(m.actual)
    print(f"\nturnos observados: {len(m.turnos)}\n")
    print(f"{'tramo':<26} {'n':>3} {'p50':>7} {'p95':>7}")
    for desde, hasta, nombre in TRAMOS:
        v = sorted(t[hasta] - t[desde] for t in m.turnos if desde in t and hasta in t)
        if not v:
            print(f"{nombre:<26} {0:>3}       -       -")
            continue
        p95 = v[int(round(0.95 * (len(v) - 1)))]
        print(f"{nombre:<26} {len(v):>3} {statistics.median(v):>6.2f}s {p95:>6.2f}s")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
