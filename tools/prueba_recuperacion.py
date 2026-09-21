"""Criterio: si cae un servidor, el resto del sistema sigue funcionando.

Se tumba cada servidor por separado y se comprueba que lo que no depende de el
sigue respondiendo, y que al volver el servidor el sistema se recupera solo.
"""
import json
import subprocess
import time
import urllib.request

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from coramo_msgs.msg import BodyCommand, Transcript

HABLA = "http://127.0.0.1:8092/say"


def servicio(accion: str, nombre: str) -> None:
    subprocess.run(["sudo", "-S", "-p", "", "systemctl", accion, nombre],
                   input="coramo123\n", text=True, check=True,
                   capture_output=True)


class Prueba(Node):
    def __init__(self):
        super().__init__("prueba_recuperacion")
        self.comandos: list[tuple[str, str]] = []
        self.textos: list[str] = []
        self.pub = self.create_publisher(Transcript, "/speech/text", 10)
        self.create_subscription(BodyCommand, "/body/command_safe", self._cmd, 50)
        self.create_subscription(Transcript, "/speech/text", self._tx, 50)
        self.rearmar = self.create_client(Trigger, "/body/rearm")

    def _cmd(self, m):
        self.comandos.append((m.tool, m.preset))

    def _tx(self, m):
        self.textos.append(m.text)

    def girar(self, segundos: float) -> None:
        fin = time.time() + segundos
        while rclpy.ok() and time.time() < fin:
            rclpy.spin_once(self, timeout_sec=0.05)

    def ordenar(self, texto: str, espera: float = 8.0):
        antes = len(self.comandos)
        m = Transcript()
        m.header.stamp = self.get_clock().now().to_msg()
        m.text = texto
        m.confidence = 1.0
        m.speech_end = self.get_clock().now().to_msg()
        self.pub.publish(m)
        fin = time.time() + espera
        while rclpy.ok() and time.time() < fin and len(self.comandos) == antes:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.comandos[antes:]

    def rearmar_cuerpo(self) -> None:
        if self.rearmar.wait_for_service(timeout_sec=5.0):
            fut = self.rearmar.call_async(Trigger.Request())
            fin = time.time() + 5
            while rclpy.ok() and not fut.done() and time.time() < fin:
                rclpy.spin_once(self, timeout_sec=0.05)


def decir_sin_silenciar(texto: str) -> None:
    cuerpo = json.dumps({"texto": texto}).encode()
    p = urllib.request.Request(HABLA, data=cuerpo,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(p, timeout=60) as r:
        json.load(r)


def main() -> None:
    rclpy.init()
    n = Prueba()
    n.girar(3.0)
    n.rearmar_cuerpo()
    resultados = []

    print("=== sin servidor de voz: una orden fisica debe llegar igual ===")
    servicio("stop", "coramo-voice")
    time.sleep(2)
    got = n.ordenar("coramo cierra la mano")
    ok = any(t == "mano" for t, _ in got)
    print(f"  comandos recibidos: {got}  ->  {'CUMPLE' if ok else 'NO CUMPLE'}")
    resultados.append(("servidor de voz caido", ok))
    # Y una pregunta, que si depende de la voz: no debe tumbar nada.
    n.ordenar("coramo que hora es", espera=6.0)
    vivos = subprocess.run(["bash", "-lc", "pgrep -c -f 'lib/coramo_brain' || true"],
                           capture_output=True, text=True).stdout.strip()
    print(f"  nodos del cerebro vivos tras la pregunta: {vivos}")
    resultados.append(("los nodos sobreviven a una respuesta sin voz",
                       int(vivos or 0) >= 6))
    servicio("start", "coramo-voice")
    time.sleep(6)

    print("\n=== sin modelo de lenguaje: la parada debe seguir siendo inmediata ===")
    servicio("stop", "coramo-llm")
    time.sleep(2)
    got = n.ordenar("coramo para")
    ok = any(t == "detener" for t, _ in got)
    print(f"  comandos recibidos: {got}  ->  {'CUMPLE' if ok else 'NO CUMPLE'}")
    resultados.append(("parada sin modelo de lenguaje", ok))
    n.rearmar_cuerpo()
    antes = len(n.comandos)
    n.ordenar("coramo levanta el brazo", espera=15.0)
    sin_modelo = len(n.comandos) == antes
    print(f"  una orden normal sin modelo no produce comando: "
          f"{'si, falla limpio' if sin_modelo else 'NO, produjo comando'}")
    resultados.append(("orden normal falla limpio sin modelo", sin_modelo))
    servicio("start", "coramo-llm")
    time.sleep(8)

    print("\n=== sin servidor de habla: el nodo debe reconectarse solo ===")
    servicio("stop", "coramo-speech")
    time.sleep(3)
    vivos = subprocess.run(["bash", "-lc", "pgrep -c -f 'lib/coramo_brain' || true"],
                           capture_output=True, text=True).stdout.strip()
    print(f"  nodos del cerebro vivos con el habla caida: {vivos}")
    resultados.append(("los nodos sobreviven a la caida del habla",
                       int(vivos or 0) >= 6))
    servicio("start", "coramo-speech")
    for _ in range(90):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2) as r:
                json.load(r)
            break
        except Exception:
            n.girar(1.0)
    n.girar(8.0)
    antes = len(n.textos)
    decir_sin_silenciar("Coramo, esta frase comprueba que el nodo de habla "
                        "volvio a conectarse solo despues de la caida.")
    fin = time.time() + 20
    while rclpy.ok() and time.time() < fin and len(n.textos) == antes:
        rclpy.spin_once(n, timeout_sec=0.05)
    volvio = len(n.textos) > antes
    print(f"  transcripcion tras reiniciar el habla: "
          f"{'llego' if volvio else 'NO llego'}")
    if volvio:
        print(f"    «{n.textos[-1]}»")
    resultados.append(("el nodo de habla se reconecta solo", volvio))

    print("\n=== resumen ===")
    for nombre, ok in resultados:
        print(f"  {'CUMPLE    ' if ok else 'NO CUMPLE '} {nombre}")
    print("\nCRITERIO CUMPLE" if all(o for _n, o in resultados)
          else "\nCRITERIO NO CUMPLE")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
