"""Criterio: el robot no debe activarse con su propia voz.

Le hace decir veinte respuestas largas por el parlante, con el microfono real
abierto, y cuenta cuantas veces se activo por lo que oyo. Debe ser cero.

Uso: python3 prueba_autoescucha.py [cuantas]
"""
import json
import sys
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node

from coramo_msgs.msg import Event, Say, Transcript

SALUD = "http://127.0.0.1:8091/health"
FLUJO = "http://127.0.0.1:8091/events"
HABLA = "http://127.0.0.1:8092/say"

CONTROL = ("Soy Coramo, un robot humanoide modular, y esta frase la digo por el "
           "parlante para comprobar que el microfono alcanza a oirme cuando "
           "nadie lo esta silenciando.")

FRASES = [
    "Soy Coramo, un robot humanoide modular construido como trabajo de titulo "
    "en la Universidad Tecnologica Metropolitana, y puedo mover la cabeza, el "
    "brazo y cada dedo de la mano por separado cuando me lo pides en voz alta.",
    "Mi cerebro corre en un servidor con una tarjeta grafica dedicada, donde "
    "conviven el reconocimiento de voz, el modelo de lenguaje y la sintesis de "
    "habla, los tres funcionando sin salir a internet para que la respuesta "
    "llegue rapido.",
    "Los dedos se mueven con servomotores gobernados por un controlador de "
    "pulsos, y cada articulacion tiene limites definidos en un archivo de "
    "configuracion que un filtro de seguridad revisa antes de dejar pasar "
    "cualquier movimiento.",
    "Cuando escucho una orden, primero espero a que termines de hablar, luego "
    "transcribo lo que dijiste, despues decido que herramienta usar, y recien "
    "entonces publico el comando que mueve el cuerpo, todo en poco mas de un "
    "segundo.",
    "Si me dices que pares, no le pregunto nada al modelo de lenguaje: esa "
    "orden tiene un camino corto y directo, porque medio segundo de diferencia "
    "importa mucho mas cuando alguien quiere que me detenga de inmediato.",
]


class Prueba(Node):
    def __init__(self):
        super().__init__("prueba_autoescucha")
        self.activaciones: list[str] = []
        self.descartes: list[str] = []
        self.transcripciones: list[str] = []
        self.pub = self.create_publisher(Say, "/tts/say", 10)
        self.create_subscription(Event, "/coramo/event", self._ev, 100)
        self.create_subscription(Transcript, "/speech/text", self._tx, 100)

    def _ev(self, m):
        if m.name == "wake_ok":
            self.activaciones.append(m.detail)
        elif m.name == "wake_no":
            self.descartes.append(m.detail)

    def _tx(self, m):
        self.transcripciones.append(m.text)

    def girar(self, segundos: float) -> None:
        fin = time.time() + segundos
        while rclpy.ok() and time.time() < fin:
            rclpy.spin_once(self, timeout_sec=0.05)


def _control_microfono() -> bool:
    """Habla saltandose el silenciado y comprueba que el microfono lo oye.

    Sin esto la prueba se aprobaria sola con el microfono desconectado: cero
    auto-escuchas no dice nada si el sistema no estaba oyendo nada.
    """
    oido: list[str] = []
    estado = {"seguir": True}

    def escuchar() -> None:
        try:
            with urllib.request.urlopen(FLUJO, timeout=60) as r:
                while estado["seguir"]:
                    linea = r.readline()
                    if not linea:
                        break
                    if linea.startswith(b"data:"):
                        ev = json.loads(linea[5:])
                        if ev["type"] == "transcript":
                            oido.append(ev["text"])
        except Exception:
            pass

    threading.Thread(target=escuchar, daemon=True).start()
    time.sleep(1.0)
    cuerpo = json.dumps({"texto": CONTROL}).encode()
    pedido = urllib.request.Request(
        HABLA, data=cuerpo, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=60) as r:
        json.load(r)
    time.sleep(4.0)
    estado["seguir"] = False
    if oido:
        print(f"control: el microfono oye al parlante -> \u00ab{oido[0]}\u00bb\n")
        return True
    print("control: el microfono no oyo nada; la prueba no seria concluyente\n")
    return False


def mudo() -> bool:
    try:
        with urllib.request.urlopen(SALUD, timeout=2) as r:
            return bool(json.load(r)["silenciado"])
    except Exception:
        return False


def main() -> None:
    cuantas = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    if not _control_microfono():
        raise SystemExit(2)

    rclpy.init()
    n = Prueba()
    n.girar(2.0)

    for i in range(cuantas):
        m = Say()
        m.header.stamp = n.get_clock().now().to_msg()
        m.text = FRASES[i % len(FRASES)]
        m.speech_end = n.get_clock().now().to_msg()
        n.pub.publish(m)

        # Esperar a que empiece a hablar y a que termine, mirando el silenciado
        # del servidor de habla, que es justo la ventana en que se taparia a si
        # mismo. Si no llega, se sigue igual y el recuento lo delatara.
        limite = time.time() + 8
        while time.time() < limite and not mudo():
            n.girar(0.05)
        limite = time.time() + 60
        while time.time() < limite and mudo():
            n.girar(0.05)
        # Un respiro con el microfono abierto: si algo del eco quedo sonando,
        # aqui es donde el detector lo recogeria.
        n.girar(2.0)
        print(f"  {i + 1:02d}/{cuantas}  activaciones hasta ahora: "
              f"{len(n.activaciones)}", flush=True)

    n.girar(3.0)
    print(f"\nrespuestas habladas: {cuantas}")
    print(f"activaciones por auto-escucha: {len(n.activaciones)}")
    print(f"transcripciones capturadas durante las respuestas: "
          f"{len(n.transcripciones)}")
    for t in n.transcripciones:
        print(f"    oyo: «{t}»")
    for t in n.descartes:
        print(f"    descartado sin activar: «{t}»")
    print("\nCRITERIO CUMPLE" if not n.activaciones else "\nCRITERIO NO CUMPLE")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
