# 06 — Asistente de Voz Coramo

Pipeline completo de voz a voz usando whisper.cpp + llama.cpp + Piper TTS, todo corriendo en las GPUs AMD RX 580.

## Stack

| Componente | Herramienta | GPU |
|---|---|---|
| Wake word + STT | whisper.cpp small fp16 | GPU 1 (renderD130) |
| LLM | llama.cpp + Qwen 2.5 7B Q4_K_M | GPU 0 (renderD129) |
| TTS | Piper (es_ES-davefx-medium) | CPU |

## Instalación

### 1. llama.cpp con Vulkan

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j4
```

### 2. Modelo Qwen 2.5 7B Q4_K_M

El modelo está partido en 2 archivos (llama.cpp los carga automáticamente apuntando al primero):

```bash
cd ~/llama.cpp/models
wget "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
wget "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
```

### 3. Piper TTS

```bash
python3 -m venv ~/coramo-env
source ~/coramo-env/bin/activate
pip install piper-tts pathvalidate

mkdir -p ~/piper-voices && cd ~/piper-voices
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
```

## Uso

```bash
python3 ~/coramo/scripts/coramo-assistant.py
```

Al arrancar carga Qwen en GPU 0 (~2-3 min la primera vez). Cuando diga `Modelo listo.` ya puedes hablar.

**Wake words:** "coramo", "hola coramo", "hey coramo", "oye coramo"

## Servicio systemd

```bash
sudo cp ~/coramo/config/coramo-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coramo-assistant
```

## Notas técnicas

- `llama-server` carga el modelo una sola vez al inicio y queda en VRAM. Las respuestas llegan por HTTP a `localhost:8080`.
- Whisper escucha chunks de 2s para detectar la wake word. Al detectarla graba 7s de pregunta.
- Audio via PipeWire (`AUDIO_DEVICE=default`). Dispositivo de grabación: USB PnP Sound Device.
- `GGML_VK_VISIBLE_DEVICES` se usa para asignar GPUs: 0=renderD129, 1=renderD130.
- renderD130 falló con `VK_ERROR_INCOMPATIBLE_DRIVER` en versiones anteriores de Vulkan pero funciona con el driver actualizado.
- Logs de diagnóstico en `~/coramo-debug.log`.
