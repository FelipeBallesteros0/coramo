# Proyecto Coramo

Raspberry Pi 5 configurada como estación de trabajo con GPUs discretas AMD RX 580, asistente de voz con IA local y control de hardware via Arduino.

## Hardware

| Componente | Detalle |
|---|---|
| SBC | Raspberry Pi 5 Model B (8GB) |
| OS | Ubuntu 24.04.4 LTS aarch64 |
| Kernel | 6.6.70-v8-16k+ (Coreforge, GPU patched) |
| GPU x2 | AMD Radeon RX 580 2048SP (8GB VRAM cada una) |
| Expansión PCIe | Suptronics X1011 M.2 PCIe Multiplexer |
| WiFi USB | MediaTek MT7921U (antena externa) |
| Audio | USB PnP Sound Device (PCM2902) |
| Microcontrolador | Arduino Mega 2560 (CH340) vía USB |

## Asistente de Voz

Pipeline completo de voz a voz con control de hardware:

```
Micrófono → Silero VAD (CPU, detecta habla)
          → whisper medium-q5_0 (GPU 1, transcripción)
          → check "coramo" en texto → descarta si no está
          → Qwen3-8B Q4_K_M (GPU 0, LLM)
          → function calling → Arduino Mega → PCA9685 → mano robótica (5 servos)
          → Piper TTS (CPU, es_ES-davefx)
          → Altavoz
```

**Wake words:** "coramo", "hola coramo", "hey coramo", "oye coramo"

**Ejemplos:**
- *"coramo abre la mano"* → gesto: todos los dedos a 0°
- *"coramo cierra la mano"* → gesto: todos los dedos a 180°
- *"coramo mueve el índice a 90 grados"* → mover_dedo(1, 90)

**Optimizaciones de latencia:**
- Streaming LLM→TTS — empieza a hablar en cuanto termina la primera oración, sin esperar la respuesta completa.
- Silero VAD — corta la grabación en cuanto el usuario deja de hablar (~1s de silencio), eliminando las esperas fijas de 8–14s.
- Overlap transcripción+grabación — mientras se graba la continuación (VAD, CPU), la transcripción del chunk inicial (GPU 1) corre en paralelo via `ThreadPoolExecutor`, reduciendo ~1-3s de latencia.
- KV cache warmup — al arrancar, el system prompt se pre-calienta en el KV cache. Las peticiones al LLM solo procesan los tokens del usuario.
- Eliminado double request — la respuesta LLM sin tool_calls se habla directamente sin re-pedir al servidor, ahorrando ~1-2s por interacción.

## Configuración del kernel

Parámetros añadidos a `/boot/firmware/cmdline.txt`:

```
amdgpu.num_kcq=0 amdgpu.lockup_timeout=180000
```

- `amdgpu.num_kcq=0` — desactiva los async compute rings. Necesario porque las GPUs corren a PCIe x1 (multiplexor ASM1184e) — los compute rings tienen timeouts en GPU 1 sin este parámetro.
- `amdgpu.lockup_timeout=180000` — extiende el timeout del ring `gfx` a 3 minutos. Sin esto, el ring `gfx` de GPU 0 hace timeout durante inferencia LLM y el servidor llama-server cae con `ErrorDeviceLost`.

## Índice de documentación

- [01 - Conectividad de red](docs/01-red.md)
- [02 - Alimentación por DC jack](docs/02-alimentacion.md)
- [03 - GPU AMD RX 580 en RPi5](docs/03-gpu.md)
- [04 - Salida de video](docs/04-video.md)
- [05 - Whisper.cpp con GPU](docs/05-whisper.md)
- [06 - Asistente de voz](docs/06-asistente-voz.md)

## Scripts y código

| Archivo | Descripción |
|---|---|
| [`scripts/coramo-assistant.py`](scripts/coramo-assistant.py) | Asistente de voz principal |
| [`scripts/arduino.py`](scripts/arduino.py) | Comunicación serial con Arduino (mover_dedo, gesto) |
| [`arduino/coramo_servo.ino`](arduino/coramo_servo.ino) | Sketch Arduino — PCA9685, 5 servos, comandos JSON |
| [`scripts/debug_mano.py`](scripts/debug_mano.py) | Debug interactivo de hardware de la mano robótica |
| [`scripts/whisper-stream.sh`](scripts/whisper-stream.sh) | Transcripción en tiempo real |
| [`scripts/fix-firmware-zst.sh`](scripts/fix-firmware-zst.sh) | Descomprimir firmwares para kernel 6.6.x |
| [`scripts/network-check.sh`](scripts/network-check.sh) | Verificar estado de red y failover |
| [`config/coramo-assistant.service`](config/coramo-assistant.service) | Servicio systemd del asistente |

## Estado del proyecto

- [x] Conectividad de red con failover WiFi USB (fix: NM dispatcher desactiva powersave tras reconexión)
- [x] Alimentación por DC jack sin advertencias
- [x] Kernel con soporte amdgpu
- [x] Dos GPU RX 580 operativas
- [x] Salida de video por GPU
- [x] Whisper medium-q5_0 en GPU 1 (~8s transcripción)
- [x] Pipeline simplificado: VAD → Whisper → check "coramo" → LLM (sin openWakeWord)
- [x] LLM Qwen3-8B Q4_K_M con GPU 0 (llama-server --device Vulkan0, --parallel 1, --cache-ram 0)
- [x] Piper TTS en español
- [x] Function calling → mano robótica 5 dedos via Arduino Mega + PCA9685
- [x] Gestos: abre/cierra mano y control de dedo individual
- [x] Silero VAD para detección de fin de habla
- [x] amdgpu.lockup_timeout=180000 para prevenir crash de llama-server por ring gfx timeout
- [x] Overlap transcripción+grabación (ThreadPoolExecutor) para reducir latencia
- [x] KV cache warmup del system prompt al arrancar
- [x] Eliminado double request LLM en respuestas sin tool_calls
