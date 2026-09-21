# src/coramo_brain/coramo_brain/nodes/supervisor_node.py
"""Nodo delgado: convierte el flujo de eventos en el estado del robot."""
import rclpy
from rclpy.node import Node
from coramo_msgs.msg import Event, State
from coramo_brain.core.state import Maquina


class SupervisorNode(Node):
    def __init__(self):
        super().__init__("supervisor")
        self._m = Maquina()
        self._pub = self.create_publisher(State, "/coramo/state", 10)
        self.create_subscription(Event, "/coramo/event", self._al_llegar, 10)
        self._publicar()

    def _publicar(self) -> None:
        msg = State()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = self._m.estado
        self._pub.publish(msg)

    def _al_llegar(self, ev: Event) -> None:
        antes = self._m.estado
        if self._m.aplicar(ev.name, ev.detail) != antes:
            self._publicar()
            self.get_logger().info(f"{antes} -> {self._m.estado} ({ev.name})")


def main():
    rclpy.init()
    nodo = SupervisorNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
