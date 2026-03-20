# 05 - Whisper.cpp con GPU (voz a texto en tiempo real)

## Contexto

- **ROCm**: no disponible en aarch64 — descartado
- **Vulkan (RADV)**: soportado por Mesa 25.x — opción elegida
- **Nota**: el driver RADV muestra `fp16: 0` para RX 580 (Polaris), por lo que corre en fp32. Esto afecta el rendimiento de modelos cuantizados.

## Rendimiento por modelo (RX 580 + Vulkan RADV)

| Modelo | Encode time | Real-time | Tamaño |
|---|---|---|---|
| small (fp16) | **291ms** | ✅ | 487MB |
| small q8_0 | 4350ms | ❌ | 252MB |
| large-v1 (fp16) | 6500ms | ❌ | 2.9GB |
| large-v3 (fp16) | 6800ms | ❌ | 2.9GB |
| medium (fp16) | ~34000ms | ❌ | 1.5GB |

> El modelo `small` fp16 es el único viable para tiempo real. Los modelos cuantizados son paradójicamente más lentos por limitaciones del driver Vulkan con esos formatos de tensor.

## Instalación

### Dependencias

```bash
sudo apt install -y git cmake build-essential libvulkan-dev \
    glslc glslang-tools libsdl2-dev ffmpeg vulkan-tools
```

### Compilar whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp

# Con Vulkan y soporte de micrófono (SDL2)
cmake -B build \
    -DGGML_VULKAN=ON \
    -DWHISPER_SDL2=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Descargar modelos

```bash
cd ~/whisper.cpp

# Modelo recomendado para tiempo real
bash ./models/download-ggml-model.sh small

# Modelo large-v3 (para transcripción no real-time)
wget -O models/ggml-large-v3.bin \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
```

## Uso

### Transcripción en tiempo real desde micrófono

```bash
cd ~/whisper.cpp && GGML_VK_VISIBLE_DEVICES=0 ./build/bin/whisper-stream \
    -m models/ggml-small.bin \
    -l es \
    --step 2000 \
    --length 6000 \
    --beam-size 1 \
    -c 0
```

### Transcripción de archivo de audio

```bash
cd ~/whisper.cpp && GGML_VK_VISIBLE_DEVICES=0 ./build/bin/whisper-cli \
    -m models/ggml-small.bin \
    -l es \
    -f audio.wav
```

### Parámetros útiles

| Parámetro | Valor | Descripción |
|---|---|---|
| `GGML_VK_VISIBLE_DEVICES` | `0` o `1` | Seleccionar RX 580 #1 o #2 |
| `-l` | `es`, `en`, `auto` | Idioma |
| `--step` | `2000` | Ventana de procesamiento en ms |
| `--length` | `6000` | Contexto de audio en ms |
| `--beam-size` | `1` | Decodificación greedy (más rápido) |
| `-c` | `0` | ID del dispositivo de captura |

### Ver dispositivos de audio disponibles

```bash
arecord -l
```

## Verificar que usa GPU

```bash
GGML_VK_DEBUG=1 ./build/bin/whisper-cli \
    -m models/ggml-small.bin \
    -f samples/jfk.wav 2>&1 | grep -iE "vulkan|gpu|device"
```

Debe mostrar: `AMD Radeon RX 580 2048SP (RADV POLARIS10)`
