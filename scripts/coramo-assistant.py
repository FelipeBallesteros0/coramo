#!/usr/bin/env python3
"""
Coramo Voice Assistant
Pipeline: whisper (GPU 1) -> wake word -> whisper (GPU 1) -> Qwen3-8B (GPU 0) -> Piper TTS
Function calling: mover_servo(angulo) -> Arduino Mega via USB serial
Logs guardados en ~/coramo-debug.log para diagnostico post-crash.
"""

import os
import sys
import subprocess
import tempfile
import time
import re
import json
import urllib.request
import signal
import atexit
sys.path.insert(0, os.path.dirname(__file__))
import arduino

# -- Log a archivo para sobrevivir crashes -----------------------------------
_log_file = open("/home/felipe/coramo-debug.log", "w", buffering=1)

def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")

# -- Paths -------------------------------------------------------------------
WHISPER_BIN          = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL_WAKE   = os.path.expanduser("~/whisper.cpp/models/ggml-small.bin")           # rapido para wake word
WHISPER_MODEL_QUERY  = os.path.expanduser("~/whisper.cpp/models/ggml-large-v3-turbo.bin")  # mejor calidad para preguntas
LLAMA_SERVER    = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LLAMA_MODEL     = os.path.expanduser("~/llama.cpp/models/Qwen3-8B-Q4_K_M.gguf")
PIPER_BIN       = os.path.expanduser("~/coramo-env/bin/piper")
PIPER_MODEL     = os.path.expanduser("~/piper-voices/es_ES-davefx-medium.onnx")

# -- Audio device ------------------------------------------------------------
AUDIO_DEVICE    = "default"

# -- GPU assignment ----------------------------------------------------------
# GPU 0 (renderD129) -> llama-server / Qwen
# GPU 1 (renderD130) -> whisper
WHISPER_GPU_ENV = {"GGML_VK_VISIBLE_DEVICES": "1"}
LLAMA_GPU_ENV   = {"GGML_VK_VISIBLE_DEVICES": "0"}

# -- llama-server settings ---------------------------------------------------
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_URL  = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# -- Wake words --------------------------------------------------------------
WAKE_WORDS = ["coramo", "hola coramo", "hey coramo", "oye coramo"]

# -- Audio settings ----------------------------------------------------------
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK_SECONDS    = 6   # chunk largo para capturar wake word + pregunta en uno
QUESTION_SECONDS = 7   # solo si wake word llegó sin pregunta

# -- Global server process ---------------------------------------------------
server_proc = None


def start_llm_server() -> None:
    global server_proc
    env = {**os.environ, **LLAMA_GPU_ENV}
    log("Cargando modelo LLM en GPU 1...")
    server_proc = subprocess.Popen([
        LLAMA_SERVER,
        "-m", LLAMA_MODEL,
        "--n-gpu-layers", "99",
        "--ctx-size", "2048",
        "-fit", "off",
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--log-disable",
    ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(300):
        try:
            urllib.request.urlopen(f"{LLAMA_URL}/health", timeout=1)
            log("Modelo listo.")
            return
        except Exception:
            time.sleep(1)

    log("ERROR: llama-server no respondio en 300s")
    sys.exit(1)


def stop_llm_server() -> None:
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
        server_proc.wait()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mover_servo",
            "description": "Mueve el servo fisico al angulo indicado en grados",
            "parameters": {
                "type": "object",
                "properties": {
                    "angulo": {
                        "type": "integer",
                        "description": "Angulo del servo en grados, entre 0 y 180",
                    }
                },
                "required": ["angulo"],
            },
        },
    }
]

SYSTEM_MSG = (
    "Eres Coramo, un asistente de voz en espanol con control de hardware. "
    "Responde de forma concisa en maximo 3 oraciones. "
    "Solo texto plano, sin listas ni markdown. /no_think"
)


def call_tool(name: str, args: dict) -> str:
    """Ejecuta una tool y retorna el resultado como string."""
    if name == "mover_servo":
        angulo = args.get("angulo", 90)
        log(f"  [tool] mover_servo({angulo})")
        result = arduino.mover_servo(angulo)
        log(f"  [tool] respuesta arduino: {result}")
        if result.get("ok"):
            return f"Servo movido a {result['angulo']} grados correctamente."
        return f"Error al mover servo: {result.get('error', 'desconocido')}"
    return f"Tool desconocida: {name}"


def _llm_request(messages: list, stream: bool, extra: dict = None) -> dict | str:
    """Hace una peticion a llama-server. Si stream=False devuelve dict, si stream=True devuelve el objeto response."""
    payload = {"messages": messages, "temperature": 0.7, "max_tokens": 120, "stream": stream}
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    if not stream:
        return json.loads(resp.read())
    return resp


def _stream_speak(resp) -> None:
    """Lee tokens SSE del LLM, acumula oraciones y las sintetiza en cuanto estan completas."""
    sentence_buf = ""
    sentence_end = re.compile(r"[.!?…]+\s*")

    for raw_line in resp:
        line = raw_line.decode().strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except Exception:
            continue
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")
        if not token:
            continue
        sentence_buf += token

        # Sintetizar cuando hay una oracion completa
        match = sentence_end.search(sentence_buf)
        if match:
            sentence = sentence_buf[:match.end()].strip()
            sentence_buf = sentence_buf[match.end():]
            if sentence:
                log(f"  [stream] '{sentence[:60]}'")
                speak(sentence)

    # Resto sin punto final
    if sentence_buf.strip():
        log(f"  [stream] '{sentence_buf.strip()[:60]}'")
        speak(sentence_buf.strip())


def ask_llm(question: str) -> None:
    """Consulta el LLM con streaming. Habla cada oracion en cuanto esta lista."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": question},
    ]

    # Primera llamada sin stream para detectar tool calls
    log("  [llm] esperando respuesta...")
    data = _llm_request(messages, stream=False, extra={"tools": TOOLS, "tool_choice": "auto"})
    msg = data["choices"][0]["message"]
    finish = data["choices"][0]["finish_reason"]
    log(f"  [llm] finish_reason={finish}")

    if finish == "tool_calls" and msg.get("tool_calls"):
        # Ejecutar tools y luego respuesta en streaming
        messages.append(msg)
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            result_str = call_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
        log("  [llm] generando respuesta final en streaming...")
        resp = _llm_request(messages, stream=True)
        _stream_speak(resp)
    else:
        # Respuesta normal — re-pedir en streaming para hablar mientras genera
        log("  [llm] re-enviando en streaming...")
        resp = _llm_request(messages, stream=True)
        _stream_speak(resp)


def record_wav(filename: str, seconds: int) -> None:
    subprocess.run([
        "arecord", "-q",
        "-D", AUDIO_DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-d", str(seconds),
        filename,
    ], check=True)


def transcribe(audio_file: str, model: str = None) -> str:
    if model is None:
        model = WHISPER_MODEL_WAKE
    env = {**os.environ, **WHISPER_GPU_ENV}
    result = subprocess.run([
        WHISPER_BIN,
        "-m", model,
        "-f", audio_file,
        "-l", "es",
        "--no-prints",
        "-nt",
    ], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        log(f"  [whisper error] rc={result.returncode} stderr={result.stderr[:200]}")
    return result.stdout.strip()


def contains_wake_word(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", text.lower().strip())
    return any(ww in normalized for ww in WAKE_WORDS)


def extract_question(text: str) -> str:
    """Extrae el texto que viene despues de la wake word."""
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    for ww in sorted(WAKE_WORDS, key=len, reverse=True):  # mas largo primero
        if ww in normalized:
            idx = normalized.index(ww) + len(ww)
            remainder = text[idx:].strip(" ,.-\n")
            return remainder
    return ""


def speak(text: str) -> None:
    if not text:
        return
    log(f"  [speak] '{text[:80]}'")
    wav_file = tempfile.mktemp(suffix=".wav")
    txt_file = tempfile.mktemp(suffix=".txt")
    try:
        with open(txt_file, "w") as f:
            f.write(text)
        log("  [speak] sintetizando con piper...")
        with open(txt_file) as stdin:
            result = subprocess.run(
                [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", wav_file],
                stdin=stdin,
                capture_output=True,
            )
        if result.returncode != 0:
            log(f"  [piper error] rc={result.returncode} stderr={result.stderr.decode()[:200]}")
            return
        log("  [speak] reproduciendo audio...")
        subprocess.run(["aplay", "-q", wav_file], check=True)
        log("  [speak] listo.")
    finally:
        for f in (wav_file, txt_file):
            if os.path.exists(f):
                os.remove(f)


def listen_for_wake_word() -> None:
    log("Coramo escuchando... (di 'coramo' para activar)")

    while True:
        chunk_file = tempfile.mktemp(suffix=".wav")
        try:
            record_wav(chunk_file, CHUNK_SECONDS)
            text = transcribe(chunk_file)

            if text:
                log(f"  [escucha] {text}")

            if contains_wake_word(text):
                log("Wake word detectado!")

                # Intentar extraer pregunta del mismo chunk
                question_text = extract_question(text)
                log(f"  [pregunta en chunk] '{question_text}'")

                if not question_text:
                    # Wake word sola: pedir que continuen
                    speak("Dime")
                    question_file = tempfile.mktemp(suffix=".wav")
                    try:
                        log(f"Escuchando pregunta ({QUESTION_SECONDS}s)...")
                        record_wav(question_file, QUESTION_SECONDS)
                        log("Transcribiendo pregunta...")
                        question_text = transcribe(question_file, model=WHISPER_MODEL_QUERY)
                        log(f"  [transcripcion] '{question_text}'")
                    except Exception as e:
                        log(f"  [error grabando pregunta] {type(e).__name__}: {e}")
                    finally:
                        if os.path.exists(question_file):
                            os.remove(question_file)
                else:
                    # Pregunta incluida en el mismo chunk, re-transcribir con modelo mejor
                    log("Pregunta en mismo chunk, re-transcribiendo con large-v3-turbo...")
                    question_text = transcribe(chunk_file, model=WHISPER_MODEL_QUERY)
                    question_text = extract_question(question_text)
                    log(f"  [re-transcripcion] '{question_text}'")

                if question_text:
                    log(f"Pregunta: {question_text}")
                    log("Enviando al LLM...")
                    try:
                        ask_llm(question_text)
                    except Exception as e:
                        log(f"  [error en LLM] {type(e).__name__}: {e}")
                else:
                    speak("No entendi la pregunta, intentalo de nuevo.")

                log("Volviendo a escuchar...")

        except subprocess.CalledProcessError as e:
            log(f"Error de audio: {e}")
            time.sleep(1)
        except Exception as e:
            log(f"Error inesperado: {type(e).__name__}: {e}")
            time.sleep(1)
        except KeyboardInterrupt:
            log("Coramo detenido.")
            sys.exit(0)
        finally:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)


if __name__ == "__main__":
    for path, name in [
        (WHISPER_BIN,   "whisper-cli"),
        (WHISPER_MODEL_WAKE,  "whisper model (wake)"),
        (WHISPER_MODEL_QUERY, "whisper model (query)"),
        (LLAMA_SERVER,  "llama-server"),
        (LLAMA_MODEL,   "Qwen3 model"),
        (PIPER_BIN,     "piper"),
        (PIPER_MODEL,   "piper voice"),
    ]:
        if not os.path.exists(path):
            log(f"ERROR: No encontrado: {name} -> {path}")
            sys.exit(1)

    atexit.register(stop_llm_server)
    atexit.register(arduino.disconnect)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    log("Conectando Arduino...")
    if arduino.connect():
        log("Arduino listo.")
    else:
        log("ADVERTENCIA: Arduino no disponible, function calling desactivado.")

    start_llm_server()
    listen_for_wake_word()
