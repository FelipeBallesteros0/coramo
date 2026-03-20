#!/bin/bash
# Transcripción de voz en tiempo real con whisper.cpp + GPU RX 580
# Uso: ./whisper-stream.sh [idioma] [gpu_id]
#   idioma: es (defecto), en, auto, etc.
#   gpu_id: 0 (RX 580 #1) o 1 (RX 580 #2)

WHISPER_DIR="$HOME/whisper.cpp"
MODEL="$WHISPER_DIR/models/ggml-small.bin"
LANG="${1:-es}"
GPU="${2:-0}"

if [ ! -f "$MODEL" ]; then
    echo "Error: modelo no encontrado en $MODEL"
    echo "Descárgalo con: cd $WHISPER_DIR && bash models/download-ggml-model.sh small"
    exit 1
fi

GGML_VK_VISIBLE_DEVICES=$GPU "$WHISPER_DIR/build/bin/whisper-stream" \
    -m "$MODEL" \
    -l "$LANG" \
    --step 2000 \
    --length 6000 \
    --beam-size 1 \
    -c 0
