#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detener instancia previa si existe
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true

echo "[coramo] Iniciando cerebro..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up
