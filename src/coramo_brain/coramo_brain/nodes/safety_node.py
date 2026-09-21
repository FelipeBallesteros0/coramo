# src/coramo_brain/coramo_brain/nodes/safety_node.py
"""Nodo delgado: escucha /body/command, publica /body/command_safe."""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from coramo_msgs.msg import BodyCommand, Event
from coramo_brain.core import safety, tools


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety")
        self.declare_parameter("joints_yaml", "")
        ruta = self.get_parameter("joints_yaml").value
        self._filtro = safety.Filtro(tools.cargar_limites(ruta))
        self._pub = self.create_publisher(BodyCommand, "/body/command_safe", 10)
        self._ev = self.create_publisher(Event, "/coramo/event", 10)
        self.create_subscription(BodyCommand, "/body/command", self._al_llegar, 10)
        self.create_service(Trigger, "/body/estop", self._parar)
        self.create_service(Trigger, "/body/rearm", self._rearmar)
        self.get_logger().info(f"seguridad lista, limites de {ruta}")

    def _evento(self, nombre: str, detalle: str = "") -> None:
        msg = Event()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.detail = nombre, detalle
        self._ev.publish(msg)

    def _al_llegar(self, msg: BodyCommand) -> None:
        ahora = self.get_clock().now().nanoseconds / 1e9
        cmd = {"tool": msg.tool, "preset": msg.preset,
               "joint_names": list(msg.joint_names),
               "joint_positions_deg": list(msg.joint_positions_deg)}
        ok, razon = self._filtro.revisar(cmd, ahora)
        if ok:
            self._pub.publish(msg)
            self._evento("command_sent", msg.tool)
        else:
            self.get_logger().warning(f"comando rechazado: {razon}")
            self._evento("command_rejected", razon)

    def _parar(self, _req, resp):
        self._filtro.parar()
        self._evento("estop")
        resp.success, resp.message = True, "detenido"
        return resp

    def _rearmar(self, _req, resp):
        self._filtro.rearmar()
        self._evento("rearm")
        resp.success, resp.message = True, "rearmado"
        return resp


def main():
    rclpy.init()
    nodo = SafetyNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
