# src/coramo_brain/test/test_agent.py
import pytest
from coramo_brain.core import agent, tools

LIM = tools.cargar_limites("src/coramo_description/config/joints.yaml")


def test_extrae_la_herramienta_de_una_respuesta_real():
    crudo = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "mano", "arguments": '{"gesto": "cierra"}'}}]}}]}
    assert agent.leer_respuesta(crudo) == ("mano", {"gesto": "cierra"})


def test_respuesta_sin_herramienta_da_error_claro():
    with pytest.raises(agent.SinHerramienta):
        agent.leer_respuesta({"choices": [{"message": {"content": "hola"}}]})


def test_argumentos_con_json_roto_dan_error_claro():
    crudo = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "mano", "arguments": "{gesto: cierra"}}]}}]}
    with pytest.raises(agent.SinHerramienta):
        agent.leer_respuesta(crudo)


def test_el_agente_convierte_la_eleccion_en_comando():
    a = agent.Agente(agent.Grabado({"cierra la mano": ("mano", {"gesto": "cierra"})}), LIM)
    r = a.procesar("cierra la mano")
    assert r.comando["preset"] == "cierra"
    assert r.texto == ""


def test_responder_no_produce_comando():
    a = agent.Agente(agent.Grabado({"que hora es": ("responder", {"texto": "son las tres"})}), LIM)
    r = a.procesar("que hora es")
    assert r.comando is None and r.texto == "son las tres"


def test_si_el_modelo_falla_el_robot_lo_dice_en_vez_de_inventar():
    a = agent.Agente(agent.Grabado({}), LIM)
    r = a.procesar("haz algo raro")
    assert r.comando is None and "entend" in r.texto.lower()
