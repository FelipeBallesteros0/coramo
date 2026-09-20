"""Utilidades comunes de los benchmarks del hito 0."""
import statistics, time
from pathlib import Path

ORDENES = Path(__file__).with_name("ordenes.txt")
DATOS = Path.home() / "datos" / "ordenes"

def cargar_ordenes():
    """Devuelve [(indice, texto, tool_esperada)] a partir de ordenes.txt."""
    filas = []
    for i, linea in enumerate(ORDENES.read_text(encoding="utf-8").splitlines(), 1):
        texto, tool = linea.rsplit("|", 1)
        filas.append((i, texto.strip(), tool.strip()))
    return filas

def medir(fn, items, calentamiento=2):
    """Ejecuta fn(item) sobre items; ignora los primeros `calentamiento`.
    Devuelve (p50, p95, resultados) en segundos."""
    tiempos, resultados = [], []
    for k, item in enumerate(items):
        t0 = time.perf_counter(); r = fn(item); dt = time.perf_counter() - t0
        if k >= calentamiento:
            tiempos.append(dt)
        resultados.append((item, dt, r))
    tiempos.sort()
    p50 = statistics.median(tiempos)
    p95 = tiempos[int(round(0.95 * (len(tiempos) - 1)))]
    return p50, p95, resultados

def imprimir(nombre, p50, p95, extra=""):
    print(f"{nombre}: p50 {p50:.2f} s | p95 {p95:.2f} s {extra}")
