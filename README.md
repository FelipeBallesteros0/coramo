# Proyecto Coramo

Raspberry Pi 5 configurada como estación de trabajo con GPU discreta AMD RX 580, transcripción de voz en tiempo real y alimentación por DC jack.

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

## Índice de documentación

- [01 - Conectividad de red](docs/01-red.md)
- [02 - Alimentación por DC jack](docs/02-alimentacion.md)
- [03 - GPU AMD RX 580 en RPi5](docs/03-gpu.md)
- [04 - Salida de video](docs/04-video.md)
- [05 - Whisper.cpp con GPU](docs/05-whisper.md)

## Scripts de utilidad

- [`scripts/whisper-stream.sh`](scripts/whisper-stream.sh) — Transcripción de voz en tiempo real
- [`scripts/fix-firmware-zst.sh`](scripts/fix-firmware-zst.sh) — Descomprimir firmwares para kernel 6.6.x
- [`scripts/network-check.sh`](scripts/network-check.sh) — Verificar estado de red y failover

## Estado del proyecto

- [x] Conectividad de red con failover WiFi USB
- [x] Alimentación por DC jack sin advertencias
- [x] Kernel con soporte amdgpu
- [x] Dos GPU RX 580 operativas
- [x] Salida de video por GPU
- [x] Whisper large-v1 / small en tiempo real con GPU
- [ ] Integración de whisper en pipeline de voz
- [ ] Configuración multi-monitor
