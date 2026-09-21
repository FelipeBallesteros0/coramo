# src/coramo_brain/test/test_tools.py
import pytest
from coramo_brain.core import tools

RUTA = "src/coramo_description/config/joints.yaml"


def test_hay_cinco_herramientas():
    nombres = {t["function"]["name"] for t in tools.TOOLS}
    assert nombres == {"mano", "brazo", "cabeza", "responder", "detener"}


def test_gesto_cierra_se_traduce_a_cinco_dedos():
    cmd = tools.a_comando("mano", {"gesto": "cierra"}, tools.cargar_limites(RUTA))
    assert cmd["tool"] == "mano"
    assert cmd["preset"] == "cierra"
    assert len(cmd["joint_names"]) == 5
    assert all(v == 180 for v in cmd["joint_positions_deg"])


def test_gesto_desconocido_se_rechaza():
    with pytest.raises(tools.ComandoInvalido):
        tools.a_comando("mano", {"gesto": "saludo_vulcano"}, tools.cargar_limites(RUTA))


def test_angulo_fuera_de_limite_se_rechaza():
    with pytest.raises(tools.ComandoInvalido):
        tools.a_comando("cabeza", {"articulaciones": {"cuello_pan": 200}}, tools.cargar_limites(RUTA))


def test_articulacion_inexistente_se_rechaza():
    with pytest.raises(tools.ComandoInvalido):
        tools.a_comando("mano", {"dedos": {"tentaculo": 90}}, tools.cargar_limites(RUTA))
