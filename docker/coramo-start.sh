#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Detener instancia previa si existe
docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# Pasar Arduino al contenedor solo si está conectado
OVERRIDE_ARGS=""
if [ -c /dev/ttyUSB0 ]; then
    echo "[coramo] Arduino detectado en /dev/ttyUSB0"
    # Crear override temporal con el device
    OVERRIDE_FILE="/tmp/coramo-arduino-override.yml"
    cat > "$OVERRIDE_FILE" <<'EOF'
services:
  coramo-brain:
    devices:
      - /dev/ttyUSB0
EOF
    OVERRIDE_ARGS="-f $OVERRIDE_FILE"
else
    echo "[coramo] Arduino no conectado — iniciando sin hardware serial"
fi

echo "[coramo] Iniciando cerebro..."
docker compose -f "$COMPOSE_FILE" $OVERRIDE_ARGS up
