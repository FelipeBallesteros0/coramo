# src/coramo_brain/test/test_integracion.py
"""Camino completo sin ROS, sin GPU y sin microfono.

Simula lo que llega del servidor de habla y comprueba que sale el comando
correcto, ya validado. Es la prueba que debe estar siempre verde.
"""
import json

from coramo_brain.core import agent, safety, speech_client, tools, wake

LIM = tools.cargar_limites("src/coramo_description/config/joints.yaml")
RESPUESTAS = {
    "cierra la mano": ("mano", {"gesto": "cierra"}),
    "mira a la izquierda": ("cabeza", {"mirar": "izquierda"}),
    "que hora es": ("responder", {"texto": "No tengo reloj todavia"}),
}


def _camino(evento_sse: bytes):
    """De bytes del servidor de habla a comando validado, como en produccion."""
    eventos = speech_client.Troceador().alimentar(evento_sse)
    transcripcion = [e for e in eventos if e["type"] == "transcript"][0]
    r = wake.revisar(transcripcion["text"])
    if not r.activado:
        return None, "", "sin activacion"
    if r.es_parada:
        return {"tool": "detener", "preset": "", "joint_names": [],
                "joint_positions_deg": []}, "", "detener"
    d = agent.Agente(agent.Grabado(RESPUESTAS), LIM).procesar(r.orden)
    return d.comando, d.texto, d.herramienta


def _sse(texto: str) -> bytes:
    return b"data: " + json.dumps(
        {"type": "transcript", "text": texto, "t_speech_end": 1.0}).encode() + b"\n\n"


def test_orden_fisica_llega_validada_hasta_el_cuerpo():
    cmd, texto, herramienta = _camino(_sse("coramo cierra la mano"))
    assert herramienta == "mano" and texto == ""
    ok, razon = safety.Filtro(LIM).revisar(cmd, ahora=1.0)
    assert ok, razon
    assert cmd["joint_positions_deg"] == [180.0] * 5


def test_pregunta_produce_respuesta_hablada_y_ningun_movimiento():
    cmd, texto, herramienta = _camino(_sse("coramo que hora es"))
    assert cmd is None and herramienta == "responder" and "reloj" in texto


def test_la_parada_no_pasa_por_el_modelo():
    cmd, _texto, herramienta = _camino(_sse("coramo detente"))
    assert herramienta == "detener" and cmd["tool"] == "detener"


def test_sin_palabra_de_activacion_no_pasa_nada():
    cmd, _texto, herramienta = _camino(_sse("cierra la mano"))
    assert cmd is None and herramienta == "sin activacion"


def test_una_orden_no_grabada_hace_que_el_robot_lo_diga():
    cmd, texto, _h = _camino(_sse("coramo baila una cueca"))
    assert cmd is None and "entend" in texto.lower()
