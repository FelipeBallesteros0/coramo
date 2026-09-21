# src/coramo_brain/test/test_voice_client.py
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from coramo_brain.core import voice_client

LLAMADAS = []


class Falso(BaseHTTPRequestHandler):
    def do_POST(self):
        LLAMADAS.append(self.path)
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        cuerpo = json.dumps({"t_first_audio": 100.5, "t_done": 101.2}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, *_a):
        pass


def _servidor():
    s = HTTPServer(("127.0.0.1", 0), Falso)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def test_decir_devuelve_las_marcas_y_silencia_el_microfono():
    LLAMADAS.clear()
    s = _servidor()
    url = f"http://127.0.0.1:{s.server_port}"
    cli = voice_client.Voz(url_voz=url, url_habla=url, timeout_s=5)
    r = cli.decir("hola")
    assert r["t_first_audio"] == 100.5
    assert "/mute" in LLAMADAS and "/say" in LLAMADAS and "/unmute" in LLAMADAS
    assert LLAMADAS.index("/mute") < LLAMADAS.index("/say") < LLAMADAS.index("/unmute")


def test_si_el_servidor_de_habla_no_esta_igual_habla():
    LLAMADAS.clear()
    s = _servidor()
    cli = voice_client.Voz(url_voz=f"http://127.0.0.1:{s.server_port}",
                           url_habla="http://127.0.0.1:1", timeout_s=2)
    r = cli.decir("hola")
    assert r["t_first_audio"] == 100.5
