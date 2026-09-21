# src/coramo_brain/test/test_safety.py
import pytest
from coramo_brain.core import safety, tools

LIM = tools.cargar_limites("src/coramo_description/config/joints.yaml")


def cmd(**kw):
    base = {"tool": "mano", "preset": "cierra",
            "joint_names": ["indice"], "joint_positions_deg": [90.0]}
    base.update(kw)
    return base


def test_comando_valido_pasa():
    f = safety.Filtro(LIM)
    ok, razon = f.revisar(cmd(), ahora=1.0)
    assert ok and razon == ""


def test_angulo_fuera_de_limite_se_rechaza():
    f = safety.Filtro(LIM)
    ok, razon = f.revisar(cmd(joint_positions_deg=[500.0]), ahora=1.0)
    assert not ok and "fuera" in razon


def test_dos_comandos_muy_seguidos_se_rechaza_el_segundo():
    f = safety.Filtro(LIM)
    assert f.revisar(cmd(), ahora=1.00)[0] is True
    ok, razon = f.revisar(cmd(), ahora=1.05)
    assert not ok and "seguidos" in razon


def test_tras_parar_no_pasa_nada_hasta_rearmar():
    f = safety.Filtro(LIM)
    f.parar()
    assert f.revisar(cmd(), ahora=2.0)[0] is False
    f.rearmar()
    assert f.revisar(cmd(), ahora=3.0)[0] is True


def test_la_parada_siempre_pasa():
    f = safety.Filtro(LIM)
    f.parar()
    ok, _ = f.revisar(cmd(tool="detener", joint_names=[], joint_positions_deg=[]), ahora=2.0)
    assert ok
