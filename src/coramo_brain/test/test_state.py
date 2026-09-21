# src/coramo_brain/test/test_state.py
from coramo_brain.core.state import Maquina


def test_recorrido_de_una_orden_fisica():
    m = Maquina()
    assert m.estado == "IDLE"
    for ev, esperado in [("speech_start", "LISTENING"), ("speech_end", "THINKING"),
                         ("tool_chosen", "ACTING"), ("command_sent", "IDLE")]:
        assert m.aplicar(ev, "mano") == esperado


def test_recorrido_de_una_respuesta_hablada():
    m = Maquina()
    m.aplicar("speech_start"); m.aplicar("speech_end")
    assert m.aplicar("tool_chosen", "responder") == "SPEAKING"
    assert m.aplicar("tts_done") == "IDLE"


def test_la_parada_manda_desde_cualquier_estado():
    m = Maquina()
    m.aplicar("speech_start")
    assert m.aplicar("estop") == "STOPPED"
    assert m.aplicar("speech_start") == "STOPPED"
    assert m.aplicar("rearm") == "IDLE"


def test_un_evento_desconocido_no_cambia_el_estado():
    m = Maquina()
    m.aplicar("speech_start")
    assert m.aplicar("cualquier_cosa") == "LISTENING"
