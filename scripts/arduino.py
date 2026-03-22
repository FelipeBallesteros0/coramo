"""
Modulo de comunicacion con Arduino via USB serial.
Envia comandos JSON y lee respuestas.
"""

import serial
import json
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200
TIMEOUT     = 3.0

NOMBRES_DEDO = ["pulgar", "indice", "medio", "anular", "menique"]

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


def mover_dedo(dedo: int, angulo: int) -> dict:
    """Mueve un dedo al angulo dado (0-180). dedo: 0=pulgar ... 4=menique."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}
    dedo   = max(0, min(4, dedo))
    angulo = max(0, min(180, angulo))
    cmd = json.dumps({"dedo": dedo, "angulo": angulo}) + "\n"
    try:
        _conn.write(cmd.encode())
        resp = _conn.readline().decode().strip()
        return json.loads(resp) if resp else {"error": "sin respuesta"}
    except Exception as e:
        return {"error": str(e)}


def gesto(nombre: str) -> dict:
    """Ejecuta un gesto predefinido: 'abre' o 'cierra'."""
    global _conn
    if _conn is None or not _conn.is_open:
        if not connect():
            return {"error": "no se pudo conectar al arduino"}
    cmd = json.dumps({"gesto": nombre}) + "\n"
    try:
        _conn.write(cmd.encode())
        resp = _conn.readline().decode().strip()
        return json.loads(resp) if resp else {"error": "sin respuesta"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Conectando...")
    if connect():
        print("Abriendo mano...")
        print(gesto("abre"))
        time.sleep(2)
        print("Cerrando mano...")
        print(gesto("cierra"))
        time.sleep(2)
        print("Moviendo indice a 90...")
        print(mover_dedo(1, 90))
        disconnect()
