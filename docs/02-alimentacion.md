# 02 - Alimentación por DC jack (Suptronics X1011)

## Problema

El Raspberry Pi 5 requiere negociación USB-C PD para operar con potencia completa. Al alimentarse desde el jack DC del X1011 (que inyecta 5V directamente sobre los contactos del USB-C), el sistema muestra advertencias y limita la corriente USB a 600mA.

## Hardware

- **Placa**: Suptronics X1011 M.2 PCIe Multiplexer
- **Jack DC**: 5.5×2.1mm, centro positivo, requiere **5V ≥ 5A**
- **Conexión**: contactos pogo sobre los pads USB-C del RPi5

> ⚠️ No alimentar por DC jack Y por USB-C simultáneamente — daña el hardware.

## Solución

### 1. EEPROM del firmware (ya configurado de fábrica en algunos casos)

```bash
sudo rpi-eeprom-config --edit
# Añadir o verificar:
# PSU_MAX_CURRENT=5000
```

### 2. config.txt

Añadir a `/boot/firmware/config.txt`:

```ini
# Alimentacion via DC jack sin negociacion USB-C PD
usb_max_current_enable=1
avoid_warnings=1
```

- `PSU_MAX_CURRENT=5000` — firmware asume fuente de 5A sin PD
- `usb_max_current_enable=1` — corriente USB completa disponible
- `avoid_warnings=1` — elimina overlay de advertencia visual

## Verificación

```bash
# Sin throttling ni bajo voltaje
vcgencmd get_throttled
# Debe devolver: throttled=0x0

vcgencmd measure_volts
```
