#!/bin/bash
set -e

echo "[start] Iniciando whisper-server en GPU..."
"$WHISPER_SERVER_BIN" \
  -m "$WHISPER_MODEL" \
  --port 8081 \
  --host 127.0.0.1 \
  --convert \
  -t 4 &
WHISPER_PID=$!

echo "[start] Esperando que whisper-server este listo..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8081/ >/dev/null 2>&1; then
        echo "[start] Whisper-server listo."
        break
    fi
    sleep 1
done

exec python3 scripts/coramo-assistant.py
