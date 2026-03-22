#!/home/felipe/coramo-env/bin/python3
"""
Script de debug de hardware para la mano robotica PCA9685.
Uso: python3 scripts/debug_mano.py
"""

import serial
import time
import json
import sys

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE   = 115200
NOMBRES     = ["pulgar", "indice", "medio", "anular", "menique"]


def enviar(s, cmd: dict) -> dict:
    s.write((json.dumps(cmd) + "\n").encode())
    resp = s.readline().decode().strip()
    return json.loads(resp) if resp else {"error": "sin respuesta"}


def main():
    print(f"Conectando a {SERIAL_PORT}...")
    try:
        s = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    time.sleep(2)
    s.reset_input_buffer()
    print("Conectado.\n")

    while True:
        print("--- MENU ---")
        print("1) Abrir mano")
        print("2) Cerrar mano")
        print("3) Abrir y cerrar (ciclo)")
        print("4) Mover dedo individual")
        print("5) Recorrer todos los dedos uno a uno")
        print("q) Salir")
        op = input("> ").strip().lower()

        if op == "1":
            r = enviar(s, {"gesto": "abre"})
            print(f"  → {r}")

        elif op == "2":
            r = enviar(s, {"gesto": "cierra"})
            print(f"  → {r}")

        elif op == "3":
            print("Abriendo...")
            print(f"  → {enviar(s, {'gesto': 'abre'})}")
            time.sleep(2)
            print("Cerrando...")
            print(f"  → {enviar(s, {'gesto': 'cierra'})}")
            time.sleep(2)
            print("Abriendo...")
            print(f"  → {enviar(s, {'gesto': 'abre'})}")

        elif op == "4":
            print(f"Dedo (0=pulgar, 1=indice, 2=medio, 3=anular, 4=menique): ", end="")
            dedo = int(input().strip())
            print(f"Angulo (0-180): ", end="")
            angulo = int(input().strip())
            r = enviar(s, {"dedo": dedo, "angulo": angulo})
            print(f"  → {r}")

        elif op == "5":
            print("Abriendo mano primero...")
            enviar(s, {"gesto": "abre"})
            time.sleep(1)
            for i in range(5):
                print(f"  Cerrando {NOMBRES[i]}...")
                enviar(s, {"dedo": i, "angulo": 180})
                time.sleep(1)
                print(f"  Abriendo {NOMBRES[i]}...")
                enviar(s, {"dedo": i, "angulo": 0})
                time.sleep(1)

        elif op == "q":
            break

        print()

    s.close()
    print("Desconectado.")


if __name__ == "__main__":
    main()
