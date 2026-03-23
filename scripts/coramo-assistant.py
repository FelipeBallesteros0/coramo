#!/home/felipe/coramo-env/bin/python3
"""
Coramo Voice Assistant
Pipeline: VAD (CPU) -> whisper medium-q5_0 (GPU 1) -> check "coramo" -> Qwen3-8B (GPU 0) -> Piper TTS
Function calling: mover_dedo(dedo,angulo) / gesto(nombre) -> PCA9685 -> mano robotica
Logs guardados en ~/coramo-debug.log para diagnostico post-crash.
"""

import os
import sys
import subprocess
import tempfile
import time
import re
import json
import wave
import urllib.request
import signal
import atexit
import difflib
import numpy as np
import scipy.io.wavfile
import torch
import concurrent.futures
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps
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
WHISPER_MODEL_QUERY  = os.path.expanduser("~/whisper.cpp/models/ggml-small.bin")             # small fp32 (~5s en GPU1 con flash-attn)
LLAMA_SERVER    = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LLAMA_MODEL     = os.path.expanduser("~/llama.cpp/models/Qwen3-8B-Q4_K_M.gguf")
PIPER_BIN       = os.path.expanduser("~/coramo-env/bin/piper")
PIPER_MODEL     = os.path.expanduser("~/piper-voices/es_ES-davefx-medium.onnx")

# -- Audio device ------------------------------------------------------------
AUDIO_DEVICE    = "default"

# -- GPU assignment ----------------------------------------------------------
# llama-server usa GPU0 (Vulkan0), whisper-cli usa GPU1 (GGML_VK_VISIBLE_DEVICES=1)
WHISPER_GPU_ENV = {"GGML_VK_VISIBLE_DEVICES": "1"}
LLAMA_GPU_ENV   = {}

# -- llama-server settings ---------------------------------------------------
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_URL  = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# -- Wake words --------------------------------------------------------------
WAKE_WORDS = ["coramo", "hola coramo", "hey coramo", "oye coramo"]

# -- Audio settings ----------------------------------------------------------
SAMPLE_RATE        = 16000
CHANNELS           = 1

# -- Silero VAD settings -------------------------------------------------------
VAD_SILENCE_MS          = 1000   # ms de silencio para considerar fin de habla
VAD_MAX_SECS            = 15     # timeout maximo de seguridad
VAD_MIN_SPEECH_MS       = 200    # habla minima detectada antes de permitir corte
VAD_CHUNK_SAMPLES       = 512    # muestras por chunk (32ms a 16kHz, requerido por silero)
VAD_NO_SPEECH_TIMEOUT_MS = 3000  # si no hay habla en los primeros 3s, el usuario ya termino

# -- Global server process ---------------------------------------------------
server_proc = None

# -- Silero VAD (inicializado una vez, corre en CPU) --------------------------
_vad_model = load_silero_vad()
_vad = VADIterator(
    _vad_model,
    threshold=0.5,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=VAD_SILENCE_MS,
    speech_pad_ms=100,
)


def start_llm_server() -> None:
    global server_proc
    env = {**os.environ, **LLAMA_GPU_ENV}
    log("Cargando modelo LLM en GPU 0...")
    server_proc = subprocess.Popen([
        LLAMA_SERVER,
        "-m", LLAMA_MODEL,
        "--device", "Vulkan0",
        "--n-gpu-layers", "99",
        "--ctx-size", "2048",
        "--parallel", "1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--cache-ram", "0",
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


def warmup_llm_cache() -> None:
    """Pre-calienta el KV cache con el system prompt para que la primera pregunta sea mas rapida."""
    try:
        payload = json.dumps({
            "messages": [{"role": "system", "content": SYSTEM_MSG}],
            "max_tokens": 1,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{LLAMA_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60)
        log("KV cache pre-calentado.")
    except Exception as e:
        log(f"  [llm] warmup fallido (no critico): {e}")


def stop_llm_server() -> None:
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
        server_proc.wait()


def ensure_llm_server() -> None:
    """Verifica que llama-server este corriendo. Lo reinicia si murio."""
    try:
        urllib.request.urlopen(f"{LLAMA_URL}/health", timeout=2)
        return  # esta vivo
    except Exception:
        pass
    log("  [llm] servidor caido, reiniciando...")
    stop_llm_server()
    time.sleep(1)
    start_llm_server()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mover_dedo",
            "description": (
                "Mueve un dedo de la mano robotica a un angulo. "
                "Dedos: 0=pulgar, 1=indice, 2=medio, 3=anular, 4=menique. "
                "Usar para: 'mueve el indice a 90', 'dobla el pulgar', 'extiende el medio'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dedo":   {"type": "integer", "description": "Indice del dedo: 0=pulgar, 1=indice, 2=medio, 3=anular, 4=menique"},
                    "angulo": {"type": "integer", "description": "Angulo en grados (0=abierto, 180=cerrado)"},
                },
                "required": ["dedo", "angulo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gesto",
            "description": (
                "Ejecuta un gesto predefinido con toda la mano. "
                "Gestos disponibles: 'abre' (extiende todos los dedos), 'cierra' (cierra el puno). "
                "Usar para: 'abre la mano', 'cierra la mano', 'haz un puno', 'extiende los dedos'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "description": "Nombre del gesto: 'abre' o 'cierra'"},
                },
                "required": ["nombre"],
            },
        },
    },
]

SYSTEM_MSG = (
    "Eres CORAMO, un robot de doble brazo con una mano robotica de 5 dedos. "
    "Asistes al usuario en cualquier tarea: fisica, tecnica, conversacional o de conocimiento general. "
    "Puedes responder preguntas sobre cocina, ciencia, historia, tecnologia, recetas, matematicas y cualquier otro tema. "
    "Hablas de forma clara, natural y util. "
    "Priorizas siempre la seguridad de las personas, la proteccion del entorno y la integridad del sistema. "
    "CORAMO significa Colaborativo, Reprogramable, Autonomo y Modular. "
    "Fuiste creado por Felipe Ballesteros Leon. "
    "Siempre responde en 3 oraciones o menos, en texto plano sin listas ni markdown. /no_think"
)


def call_tool(name: str, args: dict) -> str:
    """Ejecuta una tool y retorna el resultado como string."""
    if name == "mover_dedo":
        dedo   = args.get("dedo", 0)
        angulo = args.get("angulo", 0)
        nombre = arduino.NOMBRES_DEDO[dedo] if 0 <= dedo <= 4 else str(dedo)
        log(f"  [tool] mover_dedo({nombre}, {angulo}°)")
        result = arduino.mover_dedo(dedo, angulo)
        log(f"  [tool] respuesta: {result}")
        if result.get("ok"):
            return f"Dedo {nombre} movido a {angulo} grados."
        return f"Error: {result.get('error', 'desconocido')}"

    if name == "gesto":
        nombre = args.get("nombre", "")
        log(f"  [tool] gesto({nombre})")
        result = arduino.gesto(nombre)
        log(f"  [tool] respuesta: {result}")
        if result.get("ok"):
            return f"Gesto '{nombre}' ejecutado."
        return f"Error: {result.get('error', 'desconocido')}"

    return f"Tool desconocida: {name}"


def _llm_request(messages: list, stream: bool, extra: dict = None) -> dict | str:
    """Hace una peticion a llama-server. Reintenta una vez si el servidor cae (503)."""
    payload = {"messages": messages, "temperature": 0.7, "max_tokens": 120, "stream": stream}
    if extra:
        payload.update(extra)
    data_bytes = json.dumps(payload).encode()

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                f"{LLAMA_URL}/v1/chat/completions",
                data=data_bytes,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            if not stream:
                return json.loads(resp.read())
            return resp
        except Exception as e:
            if attempt == 0:
                log(f"  [llm] reintentando tras error: {e}")
                ensure_llm_server()
                time.sleep(1)
            else:
                raise


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


def _speak_sentences(text: str) -> None:
    """Habla un texto ya completo dividiendolo en oraciones (sin re-peticion al LLM)."""
    sentence_end = re.compile(r"[.!?…]+\s*")
    buf = text.strip()
    while buf:
        match = sentence_end.search(buf)
        if match:
            sentence = buf[:match.end()].strip()
            buf = buf[match.end():]
            if sentence:
                log(f"  [speak] '{sentence[:60]}'")
                speak(sentence)
        else:
            speak(buf)
            break


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
        # Respuesta normal — usar el contenido ya obtenido, sin segunda peticion
        content = msg.get("content", "").strip()
        if content:
            _speak_sentences(content)


def concat_wav(file1: str, file2: str, output: str) -> None:
    """Concatena dos WAV del mismo formato en uno solo."""
    import wave
    with wave.open(file1) as w1, wave.open(file2) as w2:
        params = w1.getparams()
        data = w1.readframes(w1.getnframes()) + w2.readframes(w2.getnframes())
    with wave.open(output, "wb") as out:
        out.setparams(params)
        out.writeframes(data)


def record_until_silence(filename: str) -> tuple[float, bool]:
    """Graba hasta detectar silencio con Silero VAD. Retorna (duracion, habla_detectada)."""
    _vad.reset_states()
    chunk_bytes = VAD_CHUNK_SAMPLES * 2  # int16 = 2 bytes por muestra
    frames = []
    speech_detected = False
    speech_ms = 0.0
    elapsed_ms = 0.0
    max_ms = VAD_MAX_SECS * 1000

    proc = subprocess.Popen([
        "arecord", "-q",
        "-D", AUDIO_DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-t", "raw",   # sin header WAV — datos PCM puros para el VAD
    ], stdout=subprocess.PIPE)

    try:
        while elapsed_ms < max_ms:
            raw = proc.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                break
            frames.append(raw)
            elapsed_ms += 32  # 512 samples / 16000 * 1000

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            result = _vad(samples)

            if result:
                if "start" in result:
                    speech_detected = True
                    log("  [vad] habla detectada")
                if "end" in result and speech_detected and speech_ms >= VAD_MIN_SPEECH_MS:
                    log(f"  [vad] silencio detectado tras {elapsed_ms:.0f}ms")
                    break

            if speech_detected:
                speech_ms += 32

            # Si no hay habla en los primeros N ms, el usuario ya termino en el chunk anterior
            if not speech_detected and elapsed_ms >= VAD_NO_SPEECH_TIMEOUT_MS:
                log(f"  [vad] sin habla en {elapsed_ms:.0f}ms, fin de pregunta")
                break
    finally:
        proc.terminate()
        proc.wait()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    duration = elapsed_ms / 1000.0
    log(f"  [vad] grabados {duration:.1f}s (max={VAD_MAX_SECS}s)")
    return duration, speech_detected


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
        model = WHISPER_MODEL_QUERY
    env = {**os.environ, **WHISPER_GPU_ENV}
    cmd = [
        WHISPER_BIN,
        "-m", model,
        "-f", audio_file,
        "-l", "es",
        "--no-prints",
        "-nt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        log(f"  [whisper error] rc={result.returncode} stderr={result.stderr[:200]}")
    return result.stdout.strip()


def contains_wake_word(text: str) -> bool:
    import difflib
    normalized = re.sub(r"[^\w\s]", "", text.lower().strip())
    # Coincidencia exacta
    if any(ww in normalized for ww in WAKE_WORDS):
        return True
    # Coincidencia fuzzy: comparar cada palabra del texto con "coramo"
    words = normalized.split()
    for word in words:
        if len(word) >= 4 and difflib.SequenceMatcher(None, word, "coramo").ratio() >= 0.75:
            return True
    return False


def extract_question(text: str) -> str:
    """Extrae el texto entre la primera y segunda wake word (si hay repeticion)."""
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    for ww in sorted(WAKE_WORDS, key=len, reverse=True):  # mas largo primero
        if ww in normalized:
            idx = normalized.index(ww) + len(ww)
            remainder_norm = normalized[idx:].strip()
            remainder_text = text[idx:].strip(" ,.-\n")
            # Si hay una segunda wake word, cortar ahi
            for ww2 in WAKE_WORDS:
                if ww2 in remainder_norm:
                    cut = remainder_norm.index(ww2)
                    remainder_text = remainder_text[:cut].strip(" ,.-\n")
                    break
            return remainder_text
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



def _audio_has_speech(wav_file: str, min_ratio: float = 0.05) -> bool:
    """Verifica con Silero VAD si el WAV contiene habla real (evita mandar ruido a Whisper)."""
    rate, data = scipy.io.wavfile.read(wav_file)
    audio = data.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=rate, threshold=0.5)
    if not timestamps:
        return False
    speech_samples = sum(t["end"] - t["start"] for t in timestamps)
    return (speech_samples / len(audio)) >= min_ratio


def listen_for_wake_word() -> None:
    log("Coramo escuchando... (di 'coramo' para activar)")
    audio_file = tempfile.mktemp(suffix=".wav")
    try:
        while True:
            try:
                _, has_speech = record_until_silence(audio_file)
                if not has_speech:
                    continue

                log("Transcribiendo...")
                full_text = transcribe(audio_file, model=WHISPER_MODEL_QUERY)
                log(f"  [transcripcion] '{full_text}'")

                if not contains_wake_word(full_text):
                    log("  [ignorado] no contiene wake word")
                    continue

                question_text = extract_question(full_text) or full_text
                question_text = question_text.strip()
                log(f"Pregunta: '{question_text}'")

                if question_text:
                    ensure_llm_server()
                    log("Enviando al LLM...")
                    try:
                        ask_llm(question_text)
                    except Exception as e:
                        log(f"  [error en LLM] {type(e).__name__}: {e}")
                        speak("Tuve un problema, intentalo de nuevo.")
                else:
                    speak("Dime")

            except Exception as e:
                log(f"  [error procesando] {type(e).__name__}: {e}")
                speak("Tuve un problema, intentalo de nuevo.")

            log("Volviendo a escuchar...")

    except KeyboardInterrupt:
        log("Coramo detenido.")
        sys.exit(0)
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)


if __name__ == "__main__":
    for path, name in [
        (WHISPER_BIN,         "whisper-cli"),
        (WHISPER_MODEL_QUERY, "whisper model"),
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
    warmup_llm_cache()
    listen_for_wake_word()
