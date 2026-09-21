# src/coramo_brain/coramo_brain/nodes/body_bridge_sim_node.py
"""Cuerpo simulado: acepta comandos y publica las articulaciones como si se movieran.

Permite desarrollar y medir el cerebro completo sin hardware. El subproyecto B
lo sustituye por el puente real al Pico, con la misma entrada.
"""
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from coramo_msgs.msg import BodyCommand


class BodyBridgeSim(Node):
    def __init__(self):
        super().__init__("body_bridge")
        self.declare_parameter("velocidad_grados_por_s", 180.0)
        self._vel = float(self.get_parameter("velocidad_grados_por_s").value)
        self._actual: dict[str, float] = {}
        self._objetivo: dict[str, float] = {}
        self._pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(BodyCommand, "/body/command_safe", self._al_llegar, 10)
        self._t = self.get_clock().now().nanoseconds / 1e9
        self.create_timer(0.05, self._tick)
        self.get_logger().info("cuerpo SIMULADO listo")

    def _al_llegar(self, msg: BodyCommand) -> None:
        if msg.tool == "detener":
            self._objetivo = dict(self._actual)
            self.get_logger().info("parada: se congela la posicion")
            return
        for nombre, grados in zip(msg.joint_names, msg.joint_positions_deg):
            self._objetivo[nombre] = float(grados)
            self._actual.setdefault(nombre, 0.0)

    def _tick(self) -> None:
        ahora = self.get_clock().now().nanoseconds / 1e9
        paso = self._vel * (ahora - self._t)
        self._t = ahora
        for nombre, destino in self._objetivo.items():
            actual = self._actual.get(nombre, 0.0)
            if abs(destino - actual) <= paso:
                self._actual[nombre] = destino
            else:
                self._actual[nombre] = actual + math.copysign(paso, destino - actual)
        if not self._actual:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = sorted(self._actual)
        msg.position = [math.radians(self._actual[n]) for n in msg.name]
        self._pub.publish(msg)


def main():
    rclpy.init()
    nodo = BodyBridgeSim()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
