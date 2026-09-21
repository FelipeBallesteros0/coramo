#!/bin/bash
# Aceptacion del subproyecto A: reproduce las tres comprobaciones del spec, §11.
#
#   ./tools/correr_aceptacion.sh latencias    las 30 ordenes grabadas
#   ./tools/correr_aceptacion.sh autoescucha  veinte respuestas largas al aire
#   ./tools/correr_aceptacion.sh recuperacion tumba cada servidor por turno
#
# Necesita los tres servicios instalados (coramo-llm, coramo-speech,
# coramo-voice) y el espacio de trabajo compilado.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICIO=/etc/systemd/system/coramo-speech.service

sudo_() { sudo -p 'clave: ' "$@"; }

fuente() {
    sudo_ sed -i "s|^Environment=CORAMO_FUENTE=.*|Environment=CORAMO_FUENTE=$1|" "$SERVICIO"
    sudo_ sed -i "s|^Environment=CORAMO_REPETIR=.*|Environment=CORAMO_REPETIR=$2|" "$SERVICIO"
    sudo_ systemctl daemon-reload
}

esperar_habla() {
    for _ in $(seq 90); do
        curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "el servidor de habla no respondio" >&2
    return 1
}

levantar_cerebro() {
    # Los nodos lanzados por `ros2 launch` corren como lib/coramo_brain/<exe>,
    # no como nodes.<modulo>. Si se matan por el nombre del modulo quedan vivos,
    # se duplican con los nuevos y cada transcripcion se procesa dos veces.
    pkill -f "lib/coramo_[b]rain" 2>/dev/null || true
    sleep 2
    ros2 launch coramo_bringup cerebro.launch.py perfil:=dev-sin-robot \
        > /tmp/cerebro.log 2>&1 &
    sleep 12
    echo "nodos: $(ros2 node list 2>/dev/null | sort -u | tr '\n' ' ')"
}

source /opt/ros/lyrical/setup.bash
source "$REPO/install/setup.bash"

case "${1:-latencias}" in
latencias)
    # Una sola pasada por la carpeta de ordenes grabadas, a ritmo real.
    fuente /home/coramo/datos/ordenes 1
    sudo_ systemctl stop coramo-speech
    levantar_cerebro
    sudo_ systemctl start coramo-speech
    python3 "$REPO/tools/medir_subproyecto_a.py" 260
    ;;
autoescucha)
    # Con el microfono real: el robot habla y no debe obedecerse a si mismo.
    fuente mic 1
    sudo_ systemctl restart coramo-speech
    esperar_habla
    levantar_cerebro
    python3 "$REPO/tools/prueba_autoescucha.py" 20
    ;;
recuperacion)
    fuente mic 1
    sudo_ systemctl restart coramo-speech
    esperar_habla
    levantar_cerebro
    python3 "$REPO/tools/prueba_recuperacion.py"
    ;;
*)
    echo "uso: $0 {latencias|autoescucha|recuperacion}" >&2
    exit 2
    ;;
esac
