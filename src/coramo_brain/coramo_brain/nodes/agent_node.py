# src/coramo_brain/coramo_brain/nodes/agent_node.py
"""Nodo delgado: de transcripcion a comando o respuesta hablada."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from coramo_msgs.msg import BodyCommand, Event, Transcript
from coramo_brain.core import agent, tools, wake


class AgentNode(Node):
    def __init__(self):
        super().__init__("agent")
        self.declare_parameter("joints_yaml", "")
        self.declare_parameter("url_llm", "http://127.0.0.1:8080")
        self.declare_parameter("timeout_llm_s", 10.0)
        limites = tools.cargar_limites(self.get_parameter("joints_yaml").value)
        backend = agent.LlamaServer(self.get_parameter("url_llm").value,
                                    float(self.get_parameter("timeout_llm_s").value))
        self._agente = agent.Agente(backend, limites)
        self._cmd = self.create_publisher(BodyCommand, "/body/command", 10)
        self._say = self.create_publisher(String, "/tts/say", 10)
        self._ev = self.create_publisher(Event, "/coramo/event", 10)
        self.create_subscription(Transcript, "/speech/text", self._al_llegar, 10)
        self.get_logger().info("agente listo")

    def _evento(self, nombre: str, detalle: str = "") -> None:
        msg = Event()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.detail = nombre, detalle
        self._ev.publish(msg)

    def _publicar_comando(self, cmd: dict, speech_end) -> None:
        msg = BodyCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.tool = cmd["tool"]
        msg.preset = cmd["preset"]
        msg.joint_names = list(cmd["joint_names"])
        msg.joint_positions_deg = [float(v) for v in cmd["joint_positions_deg"]]
        msg.speech_end = speech_end
        self._cmd.publish(msg)

    def _al_llegar(self, msg: Transcript) -> None:
        r = wake.revisar(msg.text)
        if not r.activado:
            self._evento("wake_no", msg.text[:40])
            return
        self._evento("wake_ok", r.orden[:40])

        if r.es_parada:
            # No se consulta al modelo: se ahorra medio segundo donde mas importa.
            self._evento("tool_chosen", "detener")
            self._publicar_comando(
                {"tool": "detener", "preset": "", "joint_names": [], "joint_positions_deg": []},
                msg.speech_end)
            return

        d = self._agente.procesar(r.orden)
        self._evento("tool_chosen", d.herramienta)
        if d.comando is not None:
            self._publicar_comando(d.comando, msg.speech_end)
        if d.texto:
            self._say.publish(String(data=d.texto))


def main():
    rclpy.init()
    nodo = AgentNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()
