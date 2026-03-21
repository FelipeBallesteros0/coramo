"""
Modulo de comunicacion con Arduino via USB serial.
Envia comandos JSON y lee respuestas.
"""

import serial
import json
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 9600
TIMEOUT     = 3.0

_conn = None


def connect() -> bool:
    global _conn
    try:
        _conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # esperar reset del Arduino tras conectar
        _conn.reset_input_buffer()
        return True
    except serial.SerialException as e:
        print(f"[arduino] Error conectando a {SERIAL_PORT}: {e}")
        return False


def disconnect() -> None:
    global _conn
    if _conn and _conn.is_open:
        _conn.close()
        _conn = None


def mover_servo(angulo: int) -> dict:
    """Mueve el servo al angulo dado (0-180). Retorna respuesta del Arduino."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}

    angulo = max(0, min(180, angulo))
    cmd = json.dumps({"servo": angulo}) + "\n"
    try:
        _conn.write(cmd.encode())
        response_line = _conn.readline().decode().strip()
        if response_line:
            return json.loads(response_line)
        return {"error": "sin respuesta del arduino"}
    except Exception as e:
        return {"error": str(e)}


def barrer_servo(inicio: int, fin: int, repeticiones: int = 1, velocidad: int = 15) -> dict:
    """Barre el servo de inicio a fin y vuelta N veces. velocidad = ms/grado (5=rapido, 50=lento)."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}
    inicio = max(0, min(180, inicio))
    fin    = max(0, min(180, fin))
    reps   = max(1, repeticiones)
    vel    = max(1, velocidad)
    # Timeout dinamico: rango * vel * 2 (ida+vuelta) * reps + 10s margen
    rango_ms = abs(fin - inicio) * vel
    timeout_s = (rango_ms / 1000) * 2 * reps + 10
    cmd = json.dumps({"barrer": {"inicio": inicio, "fin": fin, "reps": reps, "vel": vel}}) + "\n"
    try:
        _conn.timeout = timeout_s
        _conn.write(cmd.encode())
        resp = _conn.readline().decode().strip()
        _conn.timeout = TIMEOUT
        return json.loads(resp) if resp else {"error": "sin respuesta"}
    except Exception as e:
        _conn.timeout = TIMEOUT
        return {"error": str(e)}


def oscilar_servo(minimo: int = 0, maximo: int = 180, velocidad: int = 15) -> dict:
    """Inicia oscilacion continua del servo. Usar detener_servo() para parar."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}
    cmd = json.dumps({"oscilar": {"min": minimo, "max": maximo, "vel": velocidad}}) + "\n"
    try:
        _conn.write(cmd.encode())
        resp = _conn.readline().decode().strip()
        return json.loads(resp) if resp else {"error": "sin respuesta"}
    except Exception as e:
        return {"error": str(e)}


def detener_servo() -> dict:
    """Detiene cualquier movimiento en curso del servo."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}
    cmd = json.dumps({"detener": True}) + "\n"
    try:
        _conn.write(cmd.encode())
        resp = _conn.readline().decode().strip()
        return json.loads(resp) if resp else {"error": "sin respuesta"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Prueba rapida
    print("Conectando...")
    if connect():
        for angulo in [0, 90, 180, 90]:
            print(f"Moviendo a {angulo}°...", end=" ")
            resp = mover_servo(angulo)
            print(resp)
            time.sleep(1)
        disconnect()
