# src/coramo_brain/test/test_speech_client.py
from coramo_brain.core import speech_client


def test_trocea_eventos_sse_partidos_en_varios_paquetes():
    p = speech_client.Troceador()
    assert p.alimentar(b'data: {"type": "speech_start"') == []
    eventos = p.alimentar(b', "t": 1.0}\n\ndata: {"type": "speech_end", "t": 2.0}\n\n')
    assert [e["type"] for e in eventos] == ["speech_start", "speech_end"]
    assert eventos[1]["t"] == 2.0


def test_ignora_lineas_que_no_son_datos():
    p = speech_client.Troceador()
    assert p.alimentar(b": comentario\n\n") == []


def test_json_roto_no_rompe_el_flujo():
    p = speech_client.Troceador()
    eventos = p.alimentar(b'data: {roto\n\ndata: {"type": "transcript", "text": "hola"}\n\n')
    assert [e["type"] for e in eventos] == ["transcript"]
