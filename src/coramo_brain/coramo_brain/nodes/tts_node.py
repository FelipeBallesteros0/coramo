# src/coramo_brain/coramo_brain/nodes/tts_node.py
"""Nodo delgado: escucha /tts/say y hace hablar al robot."""
import threading
import rclpy
from rclpy.node import Node
from coramo_msgs.msg import Event, Say
from coramo_brain.core.voice_client import Voz


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts")
        self.declare_parameter("url_voz", "http://127.0.0.1:8092")
        self.declare_parameter("url_habla", "http://127.0.0.1:8091")
        self._voz = Voz(self.get_parameter("url_voz").value,
                        self.get_parameter("url_habla").value)
        self._ev = self.create_publisher(Event, "/coramo/event", 10)
        self.create_subscription(Say, "/tts/say", self._al_llegar, 10)
        self._ocupado = threading.Lock()
        self.get_logger().info("voz lista")

    def _evento(self, nombre: str, detalle: str = "") -> None:
        msg = Event()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.detail = nombre, detalle
        self._ev.publish(msg)

    def _al_llegar(self, msg: Say) -> None:
        threading.Thread(target=self._hablar, args=(msg.text,), daemon=True).start()

    def _hablar(self, texto: str) -> None:
        if not self._ocupado.acquire(blocking=False):
            self.get_logger().warning("ya esta hablando; se descarta la frase nueva")
            return
        try:
            r = self._voz.decir(texto)
            if r.get("t_first_audio"):
                self._evento("tts_first_audio", texto[:40])
            self._evento("tts_done")
        except Exception as e:
            self.get_logger().error(f"no se pudo hablar: {e}")
            self._evento("error", f"tts: {e}")
        finally:
            self._ocupado.release()


def main():
    rclpy.init()
    nodo = TtsNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
