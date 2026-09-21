# src/coramo_brain/coramo_brain/nodes/speech_node.py
"""Nodo delgado: convierte el flujo del servidor de habla en temas de ROS."""
import threading
import time
import rclpy
from rclpy.node import Node
from coramo_msgs.msg import Event, Transcript
from coramo_brain.core import speech_client


def _a_tiempo_ros(segundos: float):
    from builtin_interfaces.msg import Time
    t = Time()
    t.sec = int(segundos)
    t.nanosec = int((segundos - t.sec) * 1e9)
    return t


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech")
        self.declare_parameter("url_habla", "http://127.0.0.1:8091")
        self._url = self.get_parameter("url_habla").value
        self._pub = self.create_publisher(Transcript, "/speech/text", 10)
        self._ev = self.create_publisher(Event, "/coramo/event", 10)
        threading.Thread(target=self._bucle, daemon=True).start()
        self.get_logger().info(f"escuchando el flujo de {self._url}")

    def _evento(self, nombre: str, detalle: str = "") -> None:
        msg = Event()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.detail = nombre, detalle
        self._ev.publish(msg)

    def _bucle(self) -> None:
        while rclpy.ok():
            try:
                speech_client.escuchar(self._url, self._al_evento)
            except Exception as e:
                self.get_logger().warning(f"flujo caido, reintento en 2 s: {e}")
            time.sleep(2.0)

    def _al_evento(self, ev: dict) -> None:
        tipo = ev.get("type")
        if tipo in ("speech_start", "speech_end"):
            self._evento(tipo)
        elif tipo == "transcript":
            msg = Transcript()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.text = ev.get("text", "")
            msg.confidence = float(ev.get("confidence", 0.0))
            msg.speech_end = _a_tiempo_ros(float(ev.get("t_speech_end", 0.0)))
            msg.wav_path = ev.get("wav", "")
            self._pub.publish(msg)
            self._evento("transcript", msg.text[:60])


def main():
    rclpy.init()
    nodo = SpeechNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
