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
Micrófono → openWakeWord coramo.onnx (CPU, threshold 0.97, sustain 3 frames)
          → Silero VAD (CPU, fin de habla)
          → whisper large-v3-turbo (GPU 1, transcripción)
          → Qwen3-14B Q4_K_M (GPU 0+1 dual, LLM)
          → function calling → Arduino Mega → servo
          → Piper TTS (CPU, es_ES-davefx)
          → Altavoz
```

**Wake words:** "coramo", "hola coramo", "hey coramo", "oye coramo"

**Ejemplos:**
- *"coramo pon el servo a 90 grados"* → mueve a ángulo fijo
- *"coramo mueve el motor entre 0 y 180 grados 3 veces"* → barrido ida/vuelta
- *"coramo oscila el motor lentamente"* → oscilación continua
- *"coramo detén el motor"* → para cualquier movimiento

**Optimizaciones de latencia:**
- Streaming LLM→TTS — empieza a hablar en cuanto termina la primera oración, sin esperar la respuesta completa.
- Silero VAD — corta la grabación en cuanto el usuario deja de hablar (~1s de silencio), eliminando las esperas fijas de 8–14s.
- Overlap transcripción+grabación — mientras se graba la continuación (VAD, CPU), la transcripción del chunk inicial (GPU 1) corre en paralelo via `ThreadPoolExecutor`, reduciendo ~1-3s de latencia.
- KV cache warmup — al arrancar, el system prompt se pre-calienta en el KV cache. Las peticiones al LLM solo procesan los tokens del usuario.
- Eliminado double request — la respuesta LLM sin tool_calls se habla directamente sin re-pedir al servidor, ahorrando ~1-2s por interacción.

## Configuración del kernel

Parámetros añadidos a `/boot/firmware/cmdline.txt`:

```
amdgpu.num_kcq=0 amdgpu.lockup_timeout=0
```

- `amdgpu.num_kcq=0` — desactiva los async compute rings. Necesario porque las GPUs corren a PCIe x1 (multiplexor ASM1184e) — los compute rings tienen timeouts en GPU 1 sin este parámetro.
- `amdgpu.lockup_timeout=0` — desactiva el detector de lockup del ring `gfx`. Sin esto, el ring `gfx` de GPU 0 hace timeout durante inferencia LLM, el BACO reset falla al restaurar VRAM por PCIe lento (`recover vram bo from shadow failed, r=-110`) y el sistema se cuelga.

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
| [`training/train_coramo.py`](training/train_coramo.py) | Script para entrenar modelo openWakeWord custom "coramo" en PC |
| [`training/record_samples.py`](training/record_samples.py) | Graba muestras de voz real en RPi para mejorar el modelo |
| [`training/GUIA_ENTRENAMIENTO.md`](training/GUIA_ENTRENAMIENTO.md) | Guía paso a paso para entrenar en Windows/WSL2 |
| [`scripts/test_wakeword.py`](scripts/test_wakeword.py) | Diagnóstico: muestra scores de wake word en tiempo real |
| [`scripts/capture_hard_negatives.py`](scripts/capture_hard_negatives.py) | Captura clips de audio que confunden al modelo (hard negative mining) |
| [`models/coramo.onnx`](models/coramo.onnx) | Modelo openWakeWord (generado tras entrenar, no incluido en repo) |
| [`scripts/arduino.py`](scripts/arduino.py) | Comunicación serial con Arduino |
| [`arduino/coramo_servo.ino`](arduino/coramo_servo.ino) | Sketch Arduino para control de servo |
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
- [x] Whisper large-v3-turbo en tiempo real con GPU
- [x] Asistente de voz con wake word
- [x] LLM Qwen3-14B Q4_K_M con dual-GPU (llama-server --device Vulkan0,Vulkan1)
- [x] Piper TTS en español
- [x] Function calling → control de servo via Arduino Mega
- [x] Órdenes complejas de servo: barrer (ida/vuelta N veces), oscilar (continuo), detener
- [x] Silero VAD para detección de fin de habla (reemplaza grabaciones fijas)
- [x] Wake word mejorada: whisper `--prompt "Coramo,"` + fuzzy matching (difflib, ratio ≥ 0.75)
- [x] openWakeWord con modelo custom "coramo" integrado (coramo.onnx)
- [x] Reentrenamiento con voz real + hard negative mining (558 falsos positivos capturados del cuarto)
- [x] Wake word: threshold 0.97 + sustain 3 frames consecutivos — sin filtro Whisper
- [x] amdgpu.lockup_timeout=0 para prevenir kernel panic por ring gfx timeout en GPU 0
- [x] Overlap transcripción+grabación (ThreadPoolExecutor) para reducir latencia
- [x] KV cache warmup del system prompt al arrancar
- [x] Eliminado double request LLM en respuestas sin tool_calls
