#!/usr/bin/env python3
"""
Benchmark de latencia del pipeline CORAMO.
Mide cada etapa por separado con datos reales.
Requiere: whisper-server corriendo en :8081, llama-server en :8080, piper en PIPER_BIN.
"""
import os, sys, time, json, wave, struct, tempfile, subprocess, urllib.request

WHISPER_SERVER_URL = os.environ.get("WHISPER_SERVER_URL", "http://127.0.0.1:8081")
LLAMA_URL          = f"http://{os.environ.get('LLAMA_HOST','127.0.0.1')}:{os.environ.get('LLAMA_PORT','8080')}"
PIPER_BIN          = os.environ.get("PIPER_BIN", "/usr/local/piper/piper")
PIPER_MODEL        = os.environ.get("PIPER_MODEL", "/voices/es_ES-davefx-medium.onnx")
AUDIO_DEVICE       = os.environ.get("ALSA_DEVICE", "plughw:0,0")
WHISPER_MODEL      = os.environ.get("WHISPER_MODEL", "/models/ggml-small.bin")
WHISPER_BIN        = os.environ.get("WHISPER_BIN", "/whisper.cpp/build/bin/whisper-cli")

RUNS = 3  # repeticiones por prueba

def hdr(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def avg(lst): return sum(lst)/len(lst) if lst else 0

# -- Genera WAV de tono puro (silencio con voz simulada) ---------------------
def make_test_wav(seconds=2.0, text_wav=None) -> str:
    """Genera un WAV de tono 440Hz (simula voz) o graba desde el mic."""
    tmp = tempfile.mktemp(suffix=".wav")
    rate = 16000
    n = int(rate * seconds)
    import math
    samples = [int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return tmp

# -- Graba 3 segundos del mic real -------------------------------------------
def record_mic_wav(seconds=3) -> str:
    tmp = tempfile.mktemp(suffix=".wav")
    print(f"  GRABANDO {seconds}s desde mic... habla ahora!")
    subprocess.run([
        "arecord", "-q", "-D", AUDIO_DEVICE,
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(seconds), tmp,
    ], check=True)
    return tmp

# -- Multipart helper ---------------------------------------------------------
_BOUNDARY = b"----BenchBoundary"
def _build_multipart(audio_bytes, language="es"):
    def field(name, value):
        return (b"--" + _BOUNDARY + b"\r\n"
                b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n\r\n'
                + value.encode() + b"\r\n")
    file_part = (b"--" + _BOUNDARY + b"\r\n"
                 b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                 b"Content-Type: audio/wav\r\n\r\n"
                 + audio_bytes + b"\r\n")
    return field("language", language) + field("response_format", "json") + file_part + b"--" + _BOUNDARY + b"--\r\n"

# ============================================================
# BENCHMARK 1: Whisper-server GPU (HTTP)
# ============================================================
def bench_whisper_server(wav_file):
    hdr("WHISPER-SERVER (GPU HTTP)")
    times = []
    ct = f"multipart/form-data; boundary={_BOUNDARY.decode()}"
    for i in range(RUNS):
        with open(wav_file, "rb") as f:
            body = _build_multipart(f.read())
        t0 = time.perf_counter()
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(
                    f"{WHISPER_SERVER_URL}/inference",
                    data=body,
                    headers={"Content-Type": ct},
                ),
                timeout=30,
            )
            data = json.loads(resp.read())
            ms = (time.perf_counter() - t0) * 1000
            text = data.get("text", "").strip()
            times.append(ms)
            print(f"  run {i+1}: {ms:.0f}ms → '{text[:60]}'")
        except Exception as e:
            print(f"  run {i+1}: ERROR — {e}")
    if times:
        print(f"  PROMEDIO: {avg(times):.0f}ms  MIN: {min(times):.0f}ms  MAX: {max(times):.0f}ms")
    return times

# ============================================================
# BENCHMARK 2: LLM (Qwen3 via llama-server)
# ============================================================
def bench_llm():
    hdr("LLM (llama-server Vulkan)")
    questions = [
        "di hola en dos palabras",
        "cuantos son dos mas dos",
        "que hora es",
    ]
    for question in questions:
        times = []
        tokens = []
        print(f"\n  Pregunta: '{question}'")
        for i in range(RUNS):
            payload = json.dumps({
                "messages": [{"role": "user", "content": question}],
                "temperature": 0.0,
                "max_tokens": 50,
                "stream": False,
            }).encode()
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    f"{LLAMA_URL}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
                ms = (time.perf_counter() - t0) * 1000
                text = resp["choices"][0]["message"].get("content", "")
                tok = resp.get("usage", {}).get("completion_tokens", 0)
                times.append(ms)
                tokens.append(tok)
                tps = tok / (ms/1000) if ms > 0 else 0
                print(f"  run {i+1}: {ms:.0f}ms | {tok} tokens | {tps:.1f} tok/s → '{text[:50]}'")
            except Exception as e:
                print(f"  run {i+1}: ERROR — {e}")
        if times:
            print(f"  PROMEDIO: {avg(times):.0f}ms | {avg(tokens):.0f} tokens avg")

# ============================================================
# BENCHMARK 3: LLM con tool_choice (como lo usa CORAMO)
# ============================================================
def bench_llm_tools():
    hdr("LLM + TOOLS (como CORAMO real)")
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "responder",
            "description": "Responde al usuario con texto.",
            "parameters": {
                "type": "object",
                "properties": {"texto": {"type": "string"}},
                "required": ["texto"],
            },
        },
    }]
    SYSTEM = "Eres CORAMO. Respuestas cortas: máximo 1 oración. Sin markdown. /no_think"
    question = "di hola en dos palabras"
    times = []
    print(f"\n  Pregunta: '{question}' (con tool_choice=required)")
    for i in range(RUNS):
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "tools": TOOLS,
            "tool_choice": "required",
            "temperature": 0.0,
            "max_tokens": 100,
            "stream": False,
        }).encode()
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{LLAMA_URL}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            ms = (time.perf_counter() - t0) * 1000
            msg = resp["choices"][0]["message"]
            finish = resp["choices"][0]["finish_reason"]
            tok = resp.get("usage", {}).get("completion_tokens", 0)
            times.append(ms)
            tps = tok / (ms/1000) if ms > 0 else 0
            if msg.get("tool_calls"):
                args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
                text = args.get("texto", "")
            else:
                text = msg.get("content", "")
            print(f"  run {i+1}: {ms:.0f}ms | {tok} tokens | {tps:.1f} tok/s | finish={finish} → '{text[:50]}'")
        except Exception as e:
            print(f"  run {i+1}: ERROR — {e}")
    if times:
        print(f"  PROMEDIO: {avg(times):.0f}ms")

# ============================================================
# BENCHMARK 4: Piper TTS
# ============================================================
def bench_piper():
    hdr("PIPER TTS")
    phrases = [
        "Hola.",
        "Son las tres de la tarde.",
        "No entiendo la pregunta, puedes repetirla por favor.",
    ]
    for phrase in phrases:
        times = []
        audio_dur = []
        print(f"\n  Texto: '{phrase}'")
        for i in range(RUNS):
            wav_out = tempfile.mktemp(suffix=".wav")
            txt_in  = tempfile.mktemp(suffix=".txt")
            with open(txt_in, "w") as f:
                f.write(phrase)
            t0 = time.perf_counter()
            try:
                with open(txt_in) as stdin:
                    r = subprocess.run(
                        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", wav_out],
                        stdin=stdin, capture_output=True,
                    )
                ms = (time.perf_counter() - t0) * 1000
                with wave.open(wav_out) as wf:
                    dur = wf.getnframes() / wf.getframerate() * 1000
                times.append(ms)
                audio_dur.append(dur)
                print(f"  run {i+1}: síntesis={ms:.0f}ms → audio={dur:.0f}ms ({dur/1000:.1f}s)")
            except Exception as e:
                print(f"  run {i+1}: ERROR — {e}")
            finally:
                for fp in (wav_out, txt_in):
                    if os.path.exists(fp): os.remove(fp)
        if times:
            print(f"  PROMEDIO síntesis: {avg(times):.0f}ms | audio: {avg(audio_dur):.0f}ms")

# ============================================================
# RESUMEN
# ============================================================
def print_summary(w_times, vad_silence_ms=1000):
    hdr("RESUMEN ESTIMADO — latencia percibida")
    if not w_times:
        print("  Whisper no midió (servidor no disponible)")
        whisper_ms = 0
    else:
        whisper_ms = avg(w_times)

    print(f"""
  Etapa                    Duración estimada
  ─────────────────────────────────────────
  VAD silence wait         {vad_silence_ms}ms   ← espera fija al final del habla
  Whisper (GPU server)     {whisper_ms:.0f}ms   ← medido
  LLM (tool call)          ver arriba      ← medido
  Piper TTS                ver arriba      ← medido
  aplay (playback)         tiempo real del audio

  El "silencio VAD" ({vad_silence_ms}ms) es puro overhead controlable.
  Reducir de 1000ms → 600ms ahorra 400ms garantizados.
""")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("CORAMO Pipeline Benchmark")
    print(f"Whisper: {WHISPER_SERVER_URL}")
    print(f"LLM:     {LLAMA_URL}")
    print(f"Piper:   {PIPER_BIN}")

    # Generar WAV de prueba (tono 440Hz, 3s) para Whisper
    print("\nGenerando WAV de prueba (tono 440Hz, 3s)...")
    test_wav = make_test_wav(seconds=3.0)
    print(f"  WAV: {test_wav}")

    # Si existe un WAV real grabado de la voz, usarlo para Whisper
    real_wav = "/tmp/coramo-bench-voice.wav"
    if os.path.exists(real_wav):
        print(f"  Usando WAV de voz real: {real_wav}")
        whisper_wav = real_wav
    else:
        print(f"  Tip: graba voz real con: arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 3 {real_wav}")
        whisper_wav = test_wav

    w_times = bench_whisper_server(whisper_wav)
    bench_llm()
    bench_llm_tools()
    bench_piper()
    print_summary(w_times)

    os.remove(test_wav)
    print("\nBenchmark completo.")
