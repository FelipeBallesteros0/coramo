# src/coramo_brain/test/test_wake.py
import pytest
from coramo_brain.core import wake


@pytest.mark.parametrize("texto", [
    "coramo cierra la mano",
    "Coramo, cierra la mano.",
    "coramos cierra la mano",     # confusion vista en el hito 0
    "Koramo cierra la mano",      # idem
    "hola coramo cierra la mano",
    "oye coramo mira hacia arriba",
])
def test_reconoce_la_palabra_de_activacion(texto):
    assert wake.revisar(texto).activado


@pytest.mark.parametrize("texto", [
    "como cierras la mano",
    "romo",
    "cierra la mano",
    "",
])
def test_no_se_activa_sin_la_palabra(texto):
    assert not wake.revisar(texto).activado


def test_quita_la_palabra_y_deja_la_orden():
    assert wake.revisar("hola coramo cierra la mano").orden == "cierra la mano"


@pytest.mark.parametrize("texto", [
    "coramo detente", "coramo para", "coramo alto ahi", "coramo no te muevas",
])
def test_reconoce_la_parada(texto):
    r = wake.revisar(texto)
    assert r.activado and r.es_parada


def test_una_orden_normal_no_es_parada():
    assert not wake.revisar("coramo cierra la mano").es_parada
