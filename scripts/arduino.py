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
