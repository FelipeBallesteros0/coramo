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
Micrófono → whisper small (GPU 1, wake word)
          → whisper large-v3-turbo (GPU 1, transcripción)
          → Qwen3-8B Q4_K_M (GPU 0, LLM)
          → function calling → Arduino Mega → servo
          → Piper TTS (CPU, es_ES-davefx)
          → Altavoz
```

**Wake words:** "coramo", "hola coramo", "hey coramo", "oye coramo"

**Ejemplo:** *"coramo pon el servo a 90 grados"* → mueve servo físicamente y confirma en voz.

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
| [`scripts/arduino.py`](scripts/arduino.py) | Comunicación serial con Arduino |
| [`arduino/coramo_servo.ino`](arduino/coramo_servo.ino) | Sketch Arduino para control de servo |
| [`scripts/whisper-stream.sh`](scripts/whisper-stream.sh) | Transcripción en tiempo real |
| [`scripts/fix-firmware-zst.sh`](scripts/fix-firmware-zst.sh) | Descomprimir firmwares para kernel 6.6.x |
| [`scripts/network-check.sh`](scripts/network-check.sh) | Verificar estado de red y failover |
| [`config/coramo-assistant.service`](config/coramo-assistant.service) | Servicio systemd del asistente |

## Estado del proyecto

- [x] Conectividad de red con failover WiFi USB
- [x] Alimentación por DC jack sin advertencias
- [x] Kernel con soporte amdgpu
- [x] Dos GPU RX 580 operativas
- [x] Salida de video por GPU
- [x] Whisper large-v3-turbo en tiempo real con GPU
- [x] Asistente de voz con wake word
- [x] LLM Qwen3-8B con GPU (llama-server)
- [x] Piper TTS en español
- [x] Function calling → control de servo via Arduino Mega
