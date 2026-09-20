# Hito 0: Servidor listo. Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar el Xeon y la RPi5 cabeza verificados, con ROS 2 Lyrical Luth operativo entre ambas máquinas, y una tabla medida de latencia por backend (local vs nube) que decide qué usa el cerebro de CORAMO v2.

**Architecture:** Nada de código de robot todavía: solo instalación, configuración y scripts de benchmark en `tools/bench/`. Los modelos locales se cargan en entornos `uv` con Python 3.12 (el sistema trae 3.14). Cada tarea termina con un comando de verificación y su salida esperada, y con una entrada en `docs/`. Todo se ejecuta por SSH desde WSL (`ssh coramo`), salvo lo físico (cables, monitor, tarjeta SD), que hace Felipe.

**Tech Stack:** Ubuntu 26.04, driver NVIDIA 595 (CUDA 13.2 por driver), `uv` + Python 3.12, faster-whisper (CTranslate2), Kokoro, llama.cpp con CUDA + Qwen3-8B Q5_K_M, ultralytics YOLO11n, SDK `openai` y `anthropic`, ROS 2 Lyrical Luth (Fast DDS Discovery Server, foxglove_bridge, camera_ros), NetworkManager, chrony, systemd.

**Spec:** `docs/superpowers/specs/2026-09-18-coramo-v2-arquitectura-design.md` (secciones 4, 5.4, 5.6, 6.3, 8 y 9, hito 0).

## Global Constraints

- SO en ambas máquinas: **Ubuntu 26.04** (Resolute). ROS 2: **Lyrical Luth** (`ros-lyrical-*`). No se reinstala el Xeon.
- El Xeon no tiene AVX2: **ninguna inferencia en CPU**; todo modelo local corre en la RTX 4070 SUPER (`device=cuda:0`).
- La RX 580 es solo pantalla: **el monitor se conecta siempre al HDMI de la RX 580**, nunca a la 4070. Al terminar el hito, `nvidia-smi` no debe mostrar `gnome-shell` ni `Xwayland` en la 4070.
- El servidor **nunca se suspende** (targets de sleep enmascarados) y **ningún ahorro de energía** toca la red: WiFi con powersave off y adaptadores USB de red sin autosuspend.
- Python del sistema (3.14) no se toca. Cada servidor de modelo vive en `~/venvs/<nombre>` creado con `uv venv --python 3.12`.
- Direcciones fijas: Xeon por cable **192.168.1.90/24**, cabeza RPi5 **192.168.1.91/24**, router 192.168.1.1. Discovery Server en `192.168.1.90:11811`.
- Nombres de tools del benchmark = los del spec §6.2: `mano`, `brazo`, `cabeza`, `responder`, `detener`.
- Latencias se reportan como **p50 y p95 sobre 30 peticiones**, en segundos con 2 decimales, en `docs/mediciones/2026-09-XX-hito0.md` (XX = día real de la medición).
- Claves de API solo en `~/.config/coramo/env` del Xeon (modo 600). Nunca en el repo.
- Cada tarea termina con un commit en la rama `v2-planificacion` del clon `~/coramo` **del Xeon**; el push lo decide Felipe.
- Cambios de red se hacen con el WiFi activo como respaldo, para no perder el SSH.

---

### Task 0: Acceso, clon del repo y estructura de carpetas

**Files:**
- Modify: `~/.ssh/config` (en WSL)
- Create (en el Xeon): `~/coramo` (clon), `docs/instalacion/xeon.md`, `docs/mediciones/.gitkeep`, `tools/bench/README.md`

**Interfaces:**
- Produces: alias `ssh coramo` desde WSL; ruta `~/coramo` en el Xeon con rama `v2-planificacion`; carpetas `docs/instalacion/`, `docs/mediciones/`, `tools/bench/`.

- [ ] **Step 1: Alias SSH en WSL**

```bash
cat >> ~/.ssh/config <<'CFG'
Host coramo
    HostName 192.168.1.103
    User coramo
    IdentityFile ~/.ssh/id_ed25519
CFG
ssh coramo hostname
```
Expected: `coramo`. (La IP cambia a 192.168.1.90 en la Task 2; se actualiza el alias ahí.)

- [ ] **Step 2: Clonar el repo en el Xeon y crear la rama**

```bash
ssh coramo 'git clone https://github.com/FelipeBallesteros0/coramo ~/coramo && cd ~/coramo && git checkout -b v2-planificacion'
```
Expected: termina sin error; `ssh coramo 'cd ~/coramo && git branch --show-current'` imprime `v2-planificacion`.

- [ ] **Step 3: Copiar el spec y este plan al clon del Xeon**

```bash
scp -r ~/coramo/docs/superpowers coramo:~/coramo/docs/
ssh coramo 'cd ~/coramo && mkdir -p docs/instalacion docs/mediciones tools/bench && touch docs/mediciones/.gitkeep'
```
Expected: `ssh coramo 'ls ~/coramo/docs/superpowers/specs'` lista el spec de 2026-09-18.

- [ ] **Step 4: Iniciar la bitácora de instalación**

```bash
ssh coramo 'cat > ~/coramo/docs/instalacion/xeon.md <<EOT
# Instalación del Xeon (cerebro CORAMO v2)

Inventario base (2026-09-20): ASUS P9X79 LE, Xeon E5-2697 v2, 8x8 GB DDR3-1600, RTX 4070 SUPER (01:00.0, PCIe 3.0 x16), RX 580 (02:00.0), SSD SATA 240 GB, Ubuntu 26.04.1, kernel 7.0, nvidia-driver-595-open.

Cada sección de abajo la agrega la tarea del plan que la ejecuta.
EOT
cat > ~/coramo/tools/bench/README.md <<EOT
# Benchmarks del hito 0
Scripts de medición de latencia por backend. Se ejecutan en el Xeon dentro del venv indicado en cada script. Resultados en docs/mediciones/.
EOT'
```

- [ ] **Step 5: Commit**

```bash
ssh coramo 'cd ~/coramo && git add docs tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs: bitácora de instalación del Xeon y carpeta de benchmarks"'
```

---

### Task 1: La pantalla se conecta siempre por la RX 580; la 4070 queda libre

**Files:**
- Modify: `docs/instalacion/xeon.md` (sección "Pantalla")

**Interfaces:**
- Produces: la 4070 sin procesos gráficos; criterio que usan todas las mediciones posteriores.

- [ ] **Step 1 (Felipe, físico): mover el cable HDMI** del conector de la 4070 (tarjeta del slot 1, la de arriba) al HDMI de la RX 580 (slot 2). Reiniciar sesión gráfica o el equipo.

- [ ] **Step 2: Verificar qué tarjeta tiene el monitor**

```bash
ssh coramo 'for c in /sys/class/drm/card*-*; do s=$(cat $c/status); [ "$s" = connected ] && echo "$(basename $c) -> vendor $(cat $(dirname $c)/../device/vendor 2>/dev/null || cat /sys/class/drm/$(basename $c | cut -d- -f1)/device/vendor)"; done'
```
Expected: una línea `card0-HDMI-A-1 -> vendor 0x1002` (0x1002 = AMD). Si dice `0x10de`, el cable sigue en la NVIDIA.

- [ ] **Step 3: Verificar que GNOME no usa la 4070**

```bash
ssh coramo 'nvidia-smi --query-gpu=memory.used --format=csv,noheader; nvidia-smi | grep -cE "gnome-shell|Xwayland"'
```
Expected: memoria usada `< 100 MiB` y el conteo `0`. Si sigue en `1` o más: entrar a BIOS (tecla Supr al arrancar) → Advanced → System Agent / Graphics Configuration → **Primary Display = PCIE** y **PCIe slot = slot 2** (o "PEG2"); guardar y repetir.

- [ ] **Step 4: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## Pantalla
Monitor en el HDMI de la RX 580 (card0, vendor 0x1002). nvidia-smi sin gnome-shell ni Xwayland. Ajuste de BIOS aplicado: (sí/no, anotar).
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): monitor en la RX 580, 4070 libre"'
```

---

### Task 1b: El servidor nunca duerme (sin suspensión, sin ahorro de energía en red)

Estado medido el 2026-09-20: GNOME del usuario `coramo` ya tiene `sleep-inactive-ac-type = 'nothing'` e `idle-delay = 0`; los adaptadores USB de red ya están con `power/control = on`; pero los targets de suspensión de systemd están activos (`static`), GDM puede pedir suspensión desde la pantalla de login, y el WiFi tiene `Power save: on`.

**Files:**
- Create (Xeon): `/etc/udev/rules.d/70-usb-power.rules`, `/etc/NetworkManager/conf.d/wifi-powersave-off.conf`
- Modify: `docs/instalacion/xeon.md` (sección "Energía")

**Interfaces:**
- Produces: `systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target` → `masked` ×4; `iw dev wlx90de80052ea8 get power_save` → `off`; ping por WiFi < 30 ms. La Task 2 reutiliza el archivo de NetworkManager creado aquí.

- [ ] **Step 1: Enmascarar la suspensión en systemd (bloquea cualquier petición, venga de GNOME, de GDM o de un comando)**

```bash
ssh coramo "echo coramo123 | sudo -S -p '' systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null 2>&1; systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target | tr '\n' ' '"
```
Expected: `masked masked masked masked`.

- [ ] **Step 2: GNOME del usuario y pantalla de login de GDM sin suspensión automática**

```bash
ssh coramo "export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus; gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'; gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'; gsettings set org.gnome.desktop.session idle-delay 0; gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type; echo coramo123 | sudo -S -p '' -u gdm dbus-run-session -- gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null && echo 'gdm ok' || echo 'gdm: no se pudo (systemd enmascarado ya lo cubre)'"
```
Expected: `'nothing'` y `gdm ok` (si sale el aviso alternativo, no importa: el paso 1 impide suspender igual).

- [ ] **Step 3: Regla udev para que los adaptadores USB de red nunca entren en autosuspend (heredada de v1, `docs/legado/01-red.md`)**

```bash
ssh coramo "printf 'ACTION==\"add\", SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"0e8d\", ATTRS{idProduct}==\"7961\", ATTR{power/control}=\"on\"\nACTION==\"add\", SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"0bda\", ATTRS{idProduct}==\"8153\", ATTR{power/control}=\"on\"\n' | sudo -S -p '' tee /etc/udev/rules.d/70-usb-power.rules >/dev/null <<< coramo123; echo coramo123 | sudo -S -p '' udevadm control --reload; echo coramo123 | sudo -S -p '' udevadm trigger --subsystem-match=usb; sleep 2; for d in /sys/bus/usb/devices/*; do v=\$(cat \$d/idVendor 2>/dev/null); p=\$(cat \$d/idProduct 2>/dev/null); case \"\$v:\$p\" in 0e8d:7961|0bda:8153) echo \"\$v:\$p control=\$(cat \$d/power/control)\";; esac; done"
```
Expected: `0e8d:7961 control=on` y `0bda:8153 control=on` (hoy ya están en `on`; la regla lo fija tras cada reinicio o reconexión).

- [ ] **Step 4: WiFi sin ahorro de energía (NetworkManager)**

```bash
ssh coramo "printf '[connection]\nwifi.powersave = 2\n' | sudo -S -p '' tee /etc/NetworkManager/conf.d/wifi-powersave-off.conf >/dev/null <<< coramo123; echo coramo123 | sudo -S -p '' systemctl restart NetworkManager; sleep 6; iw dev wlx90de80052ea8 get power_save"
ping -c 5 192.168.1.103 | tail -1
```
Expected: `Power save: off` y `rtt min/avg/max` con avg `< 30 ms` (hoy: 121 ms). El SSH se corta unos segundos durante el reinicio de NetworkManager; reintentar si el primer comando devuelve error de conexión.

- [ ] **Step 5: Prueba de 30 minutos sin tocar el equipo**

```bash
sleep 1800; ssh coramo 'uptime; journalctl -b --no-pager | grep -ciE "entering sleep|suspend entry"'
```
Expected: `uptime` sigue creciendo, SSH responde a la primera y el conteo es `0`.

- [ ] **Step 6: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## Energía
- systemd: sleep/suspend/hibernate/hybrid-sleep enmascarados. GNOME (usuario y GDM): sin suspensión automática, idle-delay 0.
- udev 70-usb-power.rules: adaptadores USB de red (MediaTek 0e8d:7961, Realtek 0bda:8153) con power/control=on.
- WiFi: powersave off por /etc/NetworkManager/conf.d/wifi-powersave-off.conf (ping bajó de 121 ms a X ms).
- Prueba de 30 min sin suspensión: OK.
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): sin suspensión ni ahorro de energía en red"'
```

---

### Task 2: Red por cable (RTL8153) con IP fija; WiFi sin ahorro de energía

**Files:**
- Create (Xeon): `/etc/NetworkManager/conf.d/wifi-powersave-off.conf`
- Modify: `~/.ssh/config` (WSL), `docs/instalacion/xeon.md` (sección "Red")

**Interfaces:**
- Produces: Xeon accesible en `192.168.1.90` por cable; `ssh coramo` apunta ahí.

- [ ] **Step 1 (Felipe, físico): conectar un cable Ethernet** del adaptador USB Realtek (el que tiene puerto RJ45) al router de la casa. Alternativa válida: un switch pequeño junto al robot, conectado al router.

- [ ] **Step 2: Comprobar que el enlace sube**

```bash
ssh coramo 'ip -br link show enxf8ce21123f7b'
```
Expected: `enxf8ce21123f7b UP ...` (no `DOWN`, no `NO-CARRIER`).

- [ ] **Step 3: Crear la conexión cableada con IP fija (el WiFi sigue activo)**

```bash
ssh coramo "echo coramo123 | sudo -S -p '' nmcli con add type ethernet ifname enxf8ce21123f7b con-name cable ipv4.method manual ipv4.addresses 192.168.1.90/24 ipv4.gateway 192.168.1.1 ipv4.dns '192.168.1.1 1.1.1.1' ipv4.route-metric 100 connection.autoconnect yes && echo coramo123 | sudo -S -p '' nmcli con up cable"
ping -c 3 192.168.1.90
```
Expected: `Connection successfully activated` y 3 respuestas de ping con tiempo `< 5 ms`.

- [ ] **Step 4: Apuntar el alias SSH a la IP nueva y probar**

```bash
sed -i 's/HostName 192.168.1.103/HostName 192.168.1.90/' ~/.ssh/config
ssh coramo 'ip route | grep default'
```
Expected: la ruta por defecto sale por `enxf8ce21123f7b` con `metric 100` (el WiFi tiene 600 y queda de respaldo).

- [ ] **Step 5: Confirmar que el WiFi sigue sin ahorro de energía (lo fijó la Task 1b)**

```bash
ssh coramo 'iw dev wlx90de80052ea8 get power_save'
ping -c 3 192.168.1.103
```
Expected: `Power save: off` y ping por WiFi `< 30 ms`. Si volvió a `on`, repetir el paso 4 de la Task 1b.

- [ ] **Step 6: Reservar las IPs en el router** (Felipe, en la interfaz del router 192.168.1.1): reservar `192.168.1.90` para la MAC del RTL8153 (`ssh coramo 'cat /sys/class/net/enxf8ce21123f7b/address'`) y `192.168.1.91` para la RPi5 (se obtiene en la Task 10). Evita colisiones con el DHCP.

- [ ] **Step 7: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## Red
- Cable: adaptador USB Realtek RTL8153 (enxf8ce21123f7b), conexión NM "cable", 192.168.1.90/24, gw 192.168.1.1, métrica 100. Reserva DHCP hecha en el router.
- WiFi MediaTek (wlx90de80052ea8): respaldo, métrica 600, powersave off por /etc/NetworkManager/conf.d/wifi-powersave-off.conf.
- La placa no tiene Ethernet PCI.
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): red por cable con IP fija y WiFi sin powersave"'
```

---

### Task 3: Herramientas base, `uv`, Python 3.12 y CUDA toolkit

**Files:**
- Modify: `docs/instalacion/xeon.md` (sección "Herramientas")

**Interfaces:**
- Produces: `uv` en `~/.local/bin`, Python 3.12 gestionado por `uv`, `nvcc` disponible, `espeak-ng`, `ffmpeg`, `build-essential`, `cmake`, `git-lfs`, `htop`, `nvtop`.

- [ ] **Step 1: Paquetes del sistema**

```bash
ssh coramo "echo coramo123 | sudo -S -p '' apt-get update -qq && echo coramo123 | sudo -S -p '' apt-get install -y -qq build-essential cmake git git-lfs curl wget htop nvtop espeak-ng ffmpeg libsndfile1 alsa-utils python3-venv nvidia-cuda-toolkit && nvcc --version | tail -1"
```
Expected: última línea `Cuda compilation tools, release 12.x` o `13.x`. Si `nvidia-cuda-toolkit` no existe en el archivo de 26.04, usar el repositorio de NVIDIA:
```bash
ssh coramo "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/k.deb && echo coramo123 | sudo -S -p '' dpkg -i /tmp/k.deb && echo coramo123 | sudo -S -p '' apt-get update -qq && echo coramo123 | sudo -S -p '' apt-get install -y -qq cuda-toolkit-13-2 && ls /usr/local/cuda/bin/nvcc"
```
(Si `ubuntu2604` tampoco existe todavía, usar `ubuntu2404` del mismo URL: el toolkit es solo userland y funciona con el driver 595.)

- [ ] **Step 2: Instalar `uv` y Python 3.12**

```bash
ssh coramo 'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH; uv --version && uv python install 3.12 && uv python list | grep 3.12'
```
Expected: `uv 0.x.y` y una línea `cpython-3.12.x-linux-x86_64-gnu` instalada.

- [ ] **Step 3: Entorno de verificación de CUDA**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; mkdir -p ~/venvs && uv venv ~/venvs/cuda-check --python 3.12 -q && uv pip install --python ~/venvs/cuda-check/bin/python -q torch && ~/venvs/cuda-check/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```
Expected: `2.x.y+cu12x True NVIDIA GeForce RTX 4070 SUPER`. Si `False`: el driver no está cargado (`nvidia-smi` debe funcionar) o la rueda es CPU; reinstalar con `--index-url https://download.pytorch.org/whl/cu128`.

- [ ] **Step 4: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## Herramientas
- apt: build-essential cmake git git-lfs curl wget htop nvtop espeak-ng ffmpeg libsndfile1 alsa-utils nvidia-cuda-toolkit (nvcc $(nvcc --version | tail -1 | grep -oE "release [0-9.]+")).
- uv en ~/.local/bin; Python 3.12 por uv; venvs en ~/venvs/<nombre>.
- torch CUDA verificado en ~/venvs/cuda-check.
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): herramientas base, uv, Python 3.12 y CUDA"'
```

---

### Task 4: Fuente de poder y estabilidad bajo carga

**Files:**
- Create: `tools/bench/gpu_load.py`
- Modify: `docs/instalacion/xeon.md` (sección "Fuente")

**Interfaces:**
- Produces: confirmación de que las dos GPUs pueden trabajar a la vez sin reinicios.

- [ ] **Step 1 (Felipe, físico): leer la etiqueta de la fuente** y anotar potencia total, amperios del riel de 12 V y cuántos conectores PCIe de 8 pines tiene. Mínimo aceptable: 750 W y tres conectores (dos para el adaptador de la 4070, uno para la RX 580).

- [ ] **Step 2: Escribir la carga de prueba**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/gpu_load.py <<EOT
"""Carga sostenida en la 4070 durante N segundos. Uso: python gpu_load.py 120"""
import sys, time, torch
segundos = int(sys.argv[1]) if len(sys.argv) > 1 else 60
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
t0 = time.time(); n = 0
while time.time() - t0 < segundos:
    c = a @ b; n += 1
torch.cuda.synchronize()
print(f"{n} multiplicaciones en {segundos} s; max VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
EOT'
```

- [ ] **Step 3: Correr 120 s de carga mientras se vigila potencia y errores del kernel**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; (nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv -l 10 > /tmp/power.log &) ; ~/venvs/cuda-check/bin/python ~/coramo/tools/bench/gpu_load.py 120; pkill -f "nvidia-smi --query-gpu=power.draw"; tail -4 /tmp/power.log; echo coramo123 | sudo -S -p "" journalctl -k --since "-5 min" | grep -ciE "xid|reset|pcie bus error"'
```
Expected: potencia entre 180 y 220 W, temperatura < 85 °C, y el último número `0` (sin errores Xid ni de bus). Si el equipo se reinicia o aparece un Xid: la fuente no alcanza o el adaptador de la 4070 está mal conectado; parar aquí.

- [ ] **Step 4: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## Fuente
- Etiqueta: (marca, W totales, A en 12 V, conectores PCIe).
- Prueba gpu_load.py 120 s: potencia pico (W), temperatura máx (°C), errores de kernel: 0.
EOT
cd ~/coramo && git add docs tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): fuente verificada bajo carga; script gpu_load"'
```
(Rellenar los paréntesis con los valores reales antes del commit.)

---

### Task 5: TTS local (Kokoro) y set de 30 órdenes con audio

**Files:**
- Create: `tools/bench/ordenes.txt`, `tools/bench/bench_tts.py`, `tools/bench/gen_ordenes.py`
- Create (Xeon, fuera del repo): `~/datos/ordenes/*.wav`

**Interfaces:**
- Produces: `ordenes.txt` (30 líneas, una orden por línea, con la tool esperada separada por `|`); 30 WAV de 16 kHz mono en `~/datos/ordenes/NN.wav`; función `medir(fn, n) -> (p50, p95)` reutilizada en todos los benchmarks vía `tools/bench/comun.py`.

- [ ] **Step 1: Escribir el set de órdenes**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/ordenes.txt <<EOT
coramo cierra la mano|mano
coramo abre la mano|mano
coramo haz el gesto de paz|mano
coramo haz ok con la mano|mano
coramo levanta el pulgar|mano
coramo mueve el índice a noventa grados|mano
coramo cierra solo el meñique|mano
coramo haz rock|mano
coramo saluda|brazo
coramo pon el brazo en reposo|brazo
coramo extiende el brazo|brazo
coramo levanta el brazo|brazo
coramo baja el brazo despacio|brazo
coramo mira a la izquierda|cabeza
coramo mira a la derecha|cabeza
coramo mira al frente|cabeza
coramo mírame|cabeza
coramo qué hora es|responder
coramo cómo te llamas|responder
coramo quién te construyó|responder
coramo cuéntame un chiste corto|responder
coramo qué puedes hacer|responder
coramo cuánto es doce por tres|responder
coramo buenos días|responder
coramo detente|detener
coramo para|detener
coramo alto ahí|detener
coramo no te muevas|detener
hola coramo cierra el puño|mano
oye coramo mira hacia arriba|cabeza
EOT
wc -l ~/coramo/tools/bench/ordenes.txt'
```
Expected: `30`.

- [ ] **Step 2: Módulo común de medición**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/comun.py <<EOT
"""Utilidades comunes de los benchmarks del hito 0."""
import statistics, time
from pathlib import Path

ORDENES = Path(__file__).with_name("ordenes.txt")
DATOS = Path.home() / "datos" / "ordenes"

def cargar_ordenes():
    """Devuelve [(indice, texto, tool_esperada)] a partir de ordenes.txt."""
    filas = []
    for i, linea in enumerate(ORDENES.read_text(encoding="utf-8").splitlines(), 1):
        texto, tool = linea.rsplit("|", 1)
        filas.append((i, texto.strip(), tool.strip()))
    return filas

def medir(fn, items, calentamiento=2):
    """Ejecuta fn(item) sobre items; ignora los primeros `calentamiento`.
    Devuelve (p50, p95, resultados) en segundos."""
    tiempos, resultados = [], []
    for k, item in enumerate(items):
        t0 = time.perf_counter(); r = fn(item); dt = time.perf_counter() - t0
        if k >= calentamiento:
            tiempos.append(dt)
        resultados.append((item, dt, r))
    tiempos.sort()
    p50 = statistics.median(tiempos)
    p95 = tiempos[int(round(0.95 * (len(tiempos) - 1)))]
    return p50, p95, resultados

def imprimir(nombre, p50, p95, extra=""):
    print(f"{nombre}: p50 {p50:.2f} s | p95 {p95:.2f} s {extra}")
EOT'
```

- [ ] **Step 3: Entorno TTS y script de benchmark**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; uv venv ~/venvs/tts --python 3.12 -q && uv pip install --python ~/venvs/tts/bin/python -q kokoro soundfile numpy && cat > ~/coramo/tools/bench/bench_tts.py <<EOT
"""Latencia de Kokoro en GPU: tiempo hasta el primer audio por frase. venv: ~/venvs/tts"""
import sys, time, torch, soundfile as sf, numpy as np
from kokoro import KPipeline
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir
VOZ = "ef_dora"  # voz femenina en español; alternativa em_alex
pipe = KPipeline(lang_code="e", device="cuda")
def primer_audio(item):
    _, texto, _ = item
    for _gs, _ps, audio in pipe(texto, voice=VOZ):
        return len(audio)  # el primer trozo ya se podría reproducir
p50, p95, _ = medir(primer_audio, cargar_ordenes())
imprimir("TTS local Kokoro " + VOZ, p50, p95)
EOT
~/venvs/tts/bin/python ~/coramo/tools/bench/bench_tts.py'
```
Expected: una línea `TTS local Kokoro ef_dora: p50 0.2x s | p95 0.3x s` (el traductor midió 0,19 s). Si sale `> 0.6 s`, está en CPU: revisar que `torch.cuda.is_available()` sea `True` en ese venv.

- [ ] **Step 4: Generar los 30 WAV del set (voz sintética, 16 kHz mono)**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/gen_ordenes.py <<EOT
"""Genera ~/datos/ordenes/NN.wav (16 kHz mono) con Kokoro para cada orden. venv: ~/venvs/tts"""
import sys, numpy as np, soundfile as sf, torch
from kokoro import KPipeline
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, DATOS
DATOS.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="e", device="cuda")
for i, texto, _ in cargar_ordenes():
    voz = "ef_dora" if i % 2 else "em_alex"  # alterna dos voces
    partes = [np.asarray(a) for _, _, a in pipe(texto, voice=voz)]
    audio24 = np.concatenate(partes)
    audio16 = torch.nn.functional.interpolate(torch.tensor(audio24)[None, None], scale_factor=16000/24000, mode="linear")[0, 0].numpy()
    sf.write(DATOS / f"{i:02d}.wav", audio16, 16000, subtype="PCM_16")
print("generados", len(list(DATOS.glob("*.wav"))))
EOT
~/venvs/tts/bin/python ~/coramo/tools/bench/gen_ordenes.py'
```
Expected: `generados 30`. (Estas voces sintéticas sirven para el benchmark; el set definitivo con voces reales lo pide el spec §6.4 y se graba en el subproyecto A.)

- [ ] **Step 5: Commit**

```bash
ssh coramo 'cd ~/coramo && git add tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "bench: set de 30 órdenes, módulo común y benchmark de TTS local"'
```

---

### Task 6: STT local (faster-whisper large-v3-turbo)

**Files:**
- Create: `tools/bench/bench_stt.py`

**Interfaces:**
- Consumes: `~/datos/ordenes/NN.wav`, `comun.cargar_ordenes`, `comun.medir`.
- Produces: p50/p95 de transcripción y WER aproximado sobre las 30 órdenes.

- [ ] **Step 1: Entorno STT con las librerías CUDA por pip**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; uv venv ~/venvs/stt --python 3.12 -q && uv pip install --python ~/venvs/stt/bin/python -q faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12 jiwer && ~/venvs/stt/bin/python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"'
```
Expected: `4.x.y 1`. Si `0`: exportar `LD_LIBRARY_PATH` con las rutas de `nvidia/cublas/lib` y `nvidia/cudnn/lib` del venv (el script del paso 2 lo hace solo).

- [ ] **Step 2: Script de benchmark**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/bench_stt.py <<EOT
"""Latencia y WER de faster-whisper large-v3-turbo en CUDA. venv: ~/venvs/stt"""
import os, sys, site
# rutas de cuBLAS/cuDNN instaladas por pip
for pkg in ("cublas", "cudnn"):
    for sp in site.getsitepackages():
        d = os.path.join(sp, "nvidia", pkg, "lib")
        if os.path.isdir(d):
            os.environ["LD_LIBRARY_PATH"] = d + ":" + os.environ.get("LD_LIBRARY_PATH", "")
from faster_whisper import WhisperModel
from jiwer import wer
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
modelo = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
def transcribir(item):
    i, _, _ = item
    segs, _info = modelo.transcribe(str(DATOS / f"{i:02d}.wav"), language="es", beam_size=1, vad_filter=False)
    return " ".join(s.text.strip() for s in segs)
p50, p95, res = medir(transcribir, cargar_ordenes())
ref = [t.lower() for _, t, _ in cargar_ordenes()]; hyp = [r.lower() for _, _, r in res]
imprimir("STT local whisper-turbo", p50, p95, f"| WER {wer(ref, hyp)*100:.1f} %")
for (i, t, _), _, h in res[:5]:
    print(f"  {i:02d} ref: {t} | hyp: {h}")
EOT
~/venvs/stt/bin/python ~/coramo/tools/bench/bench_stt.py'
```
Expected: `STT local whisper-turbo: p50 0.1x s | p95 0.2x s | WER < 10 %` y cinco pares ref/hyp legibles. La primera ejecución descarga el modelo (~1,6 GB).

- [ ] **Step 3: Commit**

```bash
ssh coramo 'cd ~/coramo && git add tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "bench: STT local con faster-whisper turbo"'
```

---

### Task 7: LLM local (llama.cpp CUDA + Qwen3-8B Q5_K_M) con tools

**Files:**
- Create: `tools/bench/tools_coramo.json`, `tools/bench/bench_llm_local.py`, `~/coramo/tools/bench/llama-server.sh`

**Interfaces:**
- Produces: `tools_coramo.json` (esquema de las 5 tools del spec §6.2, formato OpenAI, reutilizado por el benchmark en nube); `llama-server` escuchando en `127.0.0.1:8080`; p50/p95 y acierto de tool.

- [ ] **Step 1: Compilar llama.cpp con CUDA**

```bash
ssh coramo 'cd ~ && git clone --depth=1 https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build --config Release -j 24 --target llama-server 2>&1 | tail -1 && ls -la build/bin/llama-server'
```
Expected: `[100%] Built target llama-server` y el binario listado. (10 a 15 min en este Xeon.)

- [ ] **Step 2: Descargar el modelo**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; uv tool install -q "huggingface_hub[cli]" && mkdir -p ~/modelos && ~/.local/bin/hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q5_K_M.gguf --local-dir ~/modelos && ls -la ~/modelos/Qwen3-8B-Q5_K_M.gguf'
```
Expected: archivo de ~5,9 GB. (Si el ejecutable se llama `huggingface-cli` en vez de `hf`, usar ese.)

- [ ] **Step 3: Esquema de tools y lanzador del servidor**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/tools_coramo.json <<EOT
[
 {"type":"function","function":{"name":"mano","description":"Mueve la mano robótica: un gesto completo o dedos individuales en grados.","parameters":{"type":"object","properties":{"gesto":{"type":"string","enum":["abre","cierra","paz","ok","rock","pulgar"]},"dedos":{"type":"object","additionalProperties":{"type":"number"}}},"additionalProperties":false}}},
 {"type":"function","function":{"name":"brazo","description":"Mueve el brazo a una pose nombrada o a ángulos articulares en grados.","parameters":{"type":"object","properties":{"pose":{"type":"string","enum":["reposo","saludo","extendido","arriba","abajo"]},"articulaciones":{"type":"object","additionalProperties":{"type":"number"}}},"additionalProperties":false}}},
 {"type":"function","function":{"name":"cabeza","description":"Orienta la cabeza del robot.","parameters":{"type":"object","properties":{"mirar":{"type":"string","enum":["persona","frente","izquierda","derecha","arriba","abajo"]}},"required":["mirar"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"responder","description":"Responde por voz cuando no hay acción física.","parameters":{"type":"object","properties":{"texto":{"type":"string"}},"required":["texto"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"detener","description":"Parada inmediata de todos los motores.","parameters":{"type":"object","properties":{},"additionalProperties":false}}}
]
EOT
cat > ~/coramo/tools/bench/llama-server.sh <<EOT
#!/bin/bash
exec ~/llama.cpp/build/bin/llama-server -m ~/modelos/Qwen3-8B-Q5_K_M.gguf -ngl 999 -c 8192 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --jinja --chat-template-kwargs "{\"enable_thinking\": false}" --host 127.0.0.1 --port 8080 --parallel 1
EOT
chmod +x ~/coramo/tools/bench/llama-server.sh; (nohup ~/coramo/tools/bench/llama-server.sh > /tmp/llama.log 2>&1 &); sleep 40; curl -s http://127.0.0.1:8080/v1/models | head -c 200; echo; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```
Expected: JSON con `"id":"Qwen3-8B-Q5_K_M.gguf"` y VRAM usada de 6 a 7 GiB. Si la opción `--chat-template-kwargs` no existe en esa versión, quitarla y anteponer `/no_think` al system prompt del paso 4.

- [ ] **Step 4: Benchmark con tool obligatoria**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; uv venv ~/venvs/bench --python 3.12 -q && uv pip install --python ~/venvs/bench/bin/python -q requests openai anthropic soundfile && cat > ~/coramo/tools/bench/bench_llm_local.py <<EOT
"""Latencia y acierto de tool de Qwen3-8B en llama-server. venv: ~/venvs/bench"""
import json, sys, requests
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir
URL = "http://127.0.0.1:8080/v1/chat/completions"
TOOLS = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
def pedir(item):
    _, texto, _ = item
    r = requests.post(URL, json={"model": "x", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}],
                                 "tools": TOOLS, "tool_choice": "required", "temperature": 0, "max_tokens": 80}, timeout=60).json()
    tc = r["choices"][0]["message"].get("tool_calls") or []
    return tc[0]["function"]["name"] if tc else "(sin tool)"
p50, p95, res = medir(pedir, cargar_ordenes())
aciertos = sum(1 for (_, _, esperado), _, obtenido in res if esperado == obtenido)
imprimir("LLM local Qwen3-8B", p50, p95, f"| acierto {aciertos}/30")
for (i, t, e), dt, o in res:
    if e != o: print(f"  fallo {i:02d}: {t} -> esperado {e}, obtuvo {o}")
EOT
~/venvs/bench/bin/python ~/coramo/tools/bench/bench_llm_local.py'
```
Expected: `LLM local Qwen3-8B: p50 0.3x-0.5x s | p95 < 0.8 s | acierto >= 27/30`. Cada fallo se lista para revisar el system prompt después (en el subproyecto A, no aquí).

- [ ] **Step 5: Commit**

```bash
ssh coramo 'cd ~/coramo && git add tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "bench: LLM local Qwen3-8B con tools en llama-server"'
```

---

### Task 8: Backends en nube (STT y TTS de OpenAI; Claude Haiku 4.5 y Sonnet 5)

**Files:**
- Create: `~/.config/coramo/env` (Xeon, fuera del repo), `tools/bench/bench_nube.py`

**Interfaces:**
- Consumes: `tools_coramo.json`, `~/datos/ordenes/NN.wav`, `comun`.
- Produces: p50/p95 por backend en nube, acierto de tool, costo por orden.

- [ ] **Step 1: Claves de API (Felipe pega las suyas)**

```bash
ssh coramo 'mkdir -p ~/.config/coramo && chmod 700 ~/.config/coramo && cat > ~/.config/coramo/env <<EOT
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
EOT
chmod 600 ~/.config/coramo/env && ls -la ~/.config/coramo/env'
```
Expected: `-rw------- ... env`. (Editar el archivo con las claves reales antes del paso 3.)

- [ ] **Step 2: Script de benchmark en nube**

```bash
ssh coramo 'cat > ~/coramo/tools/bench/bench_nube.py <<EOT
"""Latencia de STT/TTS de OpenAI y de Claude con tool obligatoria. venv: ~/venvs/bench
Uso: source ~/.config/coramo/env && python bench_nube.py"""
import json, sys, time
from openai import OpenAI
from anthropic import Anthropic
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from comun import cargar_ordenes, medir, imprimir, DATOS
TOOLS_OAI = json.load(open(__file__.rsplit("/", 1)[0] + "/tools_coramo.json"))
# mismas tools en formato Anthropic (input_schema en vez de parameters), estrictas
TOOLS_ANT = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "input_schema": {**t["function"]["parameters"], "required": t["function"]["parameters"].get("required", [])},
              "strict": True} for t in TOOLS_OAI]
SYSTEM = ("Eres CORAMO, un robot humanoide. Recibes una orden hablada en español y respondes "
          "SIEMPRE con exactamente una llamada a herramienta. Si la orden no mueve nada, usa responder.")
oai, ant = OpenAI(), Anthropic()
ordenes = cargar_ordenes()

def stt(item):
    i, _, _ = item
    with open(DATOS / f"{i:02d}.wav", "rb") as f:
        return oai.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f, language="es").text
p50, p95, _ = medir(stt, ordenes); imprimir("STT nube gpt-4o-mini-transcribe", p50, p95)

def tts_primer_byte(item):
    _, texto, _ = item
    with oai.audio.speech.with_streaming_response.create(model="gpt-4o-mini-tts", voice="coral", input=texto, response_format="pcm") as r:
        for _chunk in r.iter_bytes(chunk_size=4096):
            return 1
p50, p95, _ = medir(tts_primer_byte, ordenes); imprimir("TTS nube gpt-4o-mini-tts (primer byte)", p50, p95)

def claude(modelo, extra):
    uso = {"in": 0, "out": 0, "cache": 0}
    def pedir(item):
        _, texto, _ = item
        r = ant.messages.create(model=modelo, max_tokens=200,
                                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                tools=TOOLS_ANT, tool_choice={"type": "any"},
                                messages=[{"role": "user", "content": texto}], **extra)
        uso["in"] += r.usage.input_tokens; uso["out"] += r.usage.output_tokens
        uso["cache"] += getattr(r.usage, "cache_read_input_tokens", 0) or 0
        for b in r.content:
            if b.type == "tool_use":
                return b.name
        return "(sin tool)"
    p50, p95, res = medir(pedir, ordenes)
    ac = sum(1 for (_, _, e), _, o in res if e == o)
    imprimir(f"LLM nube {modelo}", p50, p95, f"| acierto {ac}/30 | tokens in {uso['in']} (cache {uso['cache']}) out {uso['out']}")

claude("claude-haiku-4-5", {})
claude("claude-sonnet-5", {"thinking": {"type": "disabled"}})
EOT
echo listo'
```

- [ ] **Step 3: Ejecutar**

```bash
ssh coramo 'source ~/.config/coramo/env && ~/venvs/bench/bin/python ~/coramo/tools/bench/bench_nube.py'
```
Expected: cuatro líneas de resultado. Referencias para juzgar: STT nube p50 típicamente 0,5 a 1,5 s (local: 0,1x); TTS nube primer byte 0,3 a 0,8 s (local: 0,2x); Claude p50 0,5 a 1,2 s con acierto ≥ 28/30. Costo por orden con Haiku 4.5 = (in×1 + out×5)/1e6 USD; con Sonnet 5 = (in×2 + out×10)/1e6 USD, con lectura de cache más barata. Si `claude-sonnet-5` rechaza `thinking disabled`, repetir con `{"output_config": {"effort": "low"}}` en vez de `thinking`.

- [ ] **Step 4: Commit (sin claves)**

```bash
ssh coramo 'cd ~/coramo && git status --short | grep -q env && echo "OJO: no commitear claves" || (git add tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "bench: backends en nube (OpenAI STT/TTS, Claude Haiku 4.5 y Sonnet 5)")'
```

---

### Task 9: Detector de personas en la 4070

**Files:**
- Create: `tools/bench/bench_vision.py`

**Interfaces:**
- Produces: FPS y latencia por cuadro de YOLO11n a 640×480 en CUDA.

- [ ] **Step 1: Entorno y script**

```bash
ssh coramo 'export PATH=$HOME/.local/bin:$PATH; uv venv ~/venvs/vision --python 3.12 -q && uv pip install --python ~/venvs/vision/bin/python -q ultralytics && cat > ~/coramo/tools/bench/bench_vision.py <<EOT
"""FPS de YOLO11n (personas) en CUDA sobre un cuadro de 640x480. venv: ~/venvs/vision"""
import time, numpy as np, cv2
from ultralytics import YOLO
m = YOLO("yolo11n.pt")
img = cv2.resize(cv2.imread(str(__import__("ultralytics").utils.ASSETS / "bus.jpg")), (640, 480))
for _ in range(10): m.predict(img, device=0, imgsz=640, classes=[0], verbose=False)  # calentamiento
N = 200; t0 = time.perf_counter()
for _ in range(N): r = m.predict(img, device=0, imgsz=640, classes=[0], verbose=False)
dt = (time.perf_counter() - t0) / N
print(f"YOLO11n 640x480 CUDA: {1/dt:.0f} FPS | {dt*1000:.1f} ms/cuadro | personas detectadas: {len(r[0].boxes)}")
EOT
~/venvs/vision/bin/python ~/coramo/tools/bench/bench_vision.py'
```
Expected: `YOLO11n 640x480 CUDA: > 100 FPS | < 10 ms/cuadro | personas detectadas: 4` (hay 4 personas en bus.jpg). El objetivo del spec (15 FPS, < 100 ms) queda cubierto con margen.

- [ ] **Step 2: Commit**

```bash
ssh coramo 'cd ~/coramo && git add tools && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "bench: detector de personas YOLO11n en CUDA"'
```

---

### Task 10: ROS 2 Lyrical Luth en el Xeon con Discovery Server y Foxglove

**Files:**
- Create (Xeon): `/etc/systemd/system/fastdds-discovery.service`, `/etc/profile.d/coramo-ros.sh`
- Modify: `docs/instalacion/xeon.md` (sección "ROS 2")

**Interfaces:**
- Produces: `ros2` funcional en Bash de login; Discovery Server en `192.168.1.90:11811`; `foxglove_bridge` en el puerto 8765; variable `ROS_DISCOVERY_SERVER` y `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` para todos los nodos.

- [ ] **Step 1: Repositorio de ROS 2 e instalación**

```bash
ssh coramo "echo coramo123 | sudo -S -p '' apt-get install -y -qq software-properties-common curl && echo coramo123 | sudo -S -p '' add-apt-repository -y universe >/dev/null && V=\$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F'\"' '{print \$4}') && curl -sL -o /tmp/ros2-apt-source.deb \"https://github.com/ros-infrastructure/ros-apt-source/releases/download/\${V}/ros2-apt-source_\${V}.\$(. /etc/os-release && echo \$VERSION_CODENAME)_all.deb\" && echo coramo123 | sudo -S -p '' dpkg -i /tmp/ros2-apt-source.deb && echo coramo123 | sudo -S -p '' apt-get update -qq && echo coramo123 | sudo -S -p '' apt-get install -y -qq ros-lyrical-ros-base ros-dev-tools ros-lyrical-rmw-fastrtps-cpp ros-lyrical-foxglove-bridge ros-lyrical-demo-nodes-cpp && ls /opt/ros"
```
Expected: `lyrical`.

- [ ] **Step 2: Entorno de ROS para todas las sesiones**

```bash
ssh coramo "printf 'source /opt/ros/lyrical/setup.bash\nexport RMW_IMPLEMENTATION=rmw_fastrtps_cpp\nexport ROS_DISCOVERY_SERVER=192.168.1.90:11811\nexport ROS_DOMAIN_ID=7\n' | sudo -S -p '' tee /etc/profile.d/coramo-ros.sh >/dev/null <<< coramo123; grep -q coramo-ros ~/.bashrc || echo 'source /etc/profile.d/coramo-ros.sh' >> ~/.bashrc; bash -lc 'ros2 doctor --report 2>/dev/null | grep -iE \"middleware|distribution\"'"
```
Expected: `distribution name : lyrical` y `middleware name : rmw_fastrtps_cpp`.

- [ ] **Step 3: Discovery Server como servicio**

```bash
ssh coramo "printf '[Unit]\nDescription=Fast DDS Discovery Server (CORAMO)\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nExecStart=/bin/bash -lc \"source /opt/ros/lyrical/setup.bash && exec fastdds discovery -i 0 -l 192.168.1.90 -p 11811\"\nRestart=always\nRestartSec=3\nUser=coramo\n\n[Install]\nWantedBy=multi-user.target\n' | sudo -S -p '' tee /etc/systemd/system/fastdds-discovery.service >/dev/null <<< coramo123; echo coramo123 | sudo -S -p '' systemctl daemon-reload; echo coramo123 | sudo -S -p '' systemctl enable --now fastdds-discovery; sleep 2; systemctl is-active fastdds-discovery; ss -lunp | grep -c 11811"
```
Expected: `active` y `1`.

- [ ] **Step 4: Prueba talker/listener a través del Discovery Server**

```bash
ssh coramo "bash -lc '(timeout 8 ros2 run demo_nodes_cpp talker >/dev/null 2>&1 &); timeout 8 ros2 run demo_nodes_cpp listener 2>&1 | grep -c \"I heard\"'"
```
Expected: un número `>= 3`.

- [ ] **Step 5: foxglove_bridge y conexión desde Windows**

```bash
ssh coramo "bash -lc '(nohup ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 > /tmp/foxglove.log 2>&1 &); sleep 3; ss -ltnp | grep -c 8765'"
```
Expected: `1`. En Windows, abrir Foxglove Studio → Open connection → `ws://192.168.1.90:8765`; debe listar `/rosout` y `/parameter_events`.

- [ ] **Step 6: Documentar y commit**

```bash
ssh coramo 'cat >> ~/coramo/docs/instalacion/xeon.md <<EOT

## ROS 2
- Lyrical Luth (ros-lyrical-ros-base, ros-dev-tools, rmw-fastrtps-cpp, foxglove-bridge, demo-nodes-cpp) desde el repo ros2-apt-source.
- /etc/profile.d/coramo-ros.sh: setup.bash, RMW_IMPLEMENTATION=rmw_fastrtps_cpp, ROS_DISCOVERY_SERVER=192.168.1.90:11811, ROS_DOMAIN_ID=7.
- Servicio fastdds-discovery.service (puerto UDP 11811). talker/listener OK. foxglove_bridge en 8765 (lanzado a mano por ahora; en el subproyecto A pasa a launch).
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(xeon): ROS 2 Lyrical con Discovery Server y Foxglove"'
```

---

### Task 11: Cabeza RPi5 con Ubuntu 26.04, Lyrical y las dos cámaras

**Files:**
- Create: `head/README.md`, `head/config.txt.snippet`, `head/head_cameras.launch.py`, `head/head-cameras.service`, `docs/instalacion/cabeza.md`

**Interfaces:**
- Produces: RPi5 en `192.168.1.91` publicando `/head/cam_left/image_raw/compressed` y `/head/cam_right/image_raw/compressed` a 640×480 y 15 FPS, visibles desde el Xeon.

- [ ] **Step 1 (Felipe, físico): reinstalar la RPi5.** Quitar el X1011, las RX 580 y su alimentación. Con Raspberry Pi Imager en Windows: **Ubuntu Server 26.04 LTS (64-bit)** en la tarjeta o NVMe; en "personalizar": hostname `cabeza`, usuario `coramo`, contraseña `coramo123`, SSH activado con la llave pública de `~/.ssh/id_ed25519.pub`, sin WiFi. Conectar las dos cámaras CSI y un cable Ethernet al mismo router/switch que el Xeon. Arrancar.

- [ ] **Step 2: IP fija y verificación de cámaras**

```bash
ssh coramo@cabeza.local "echo coramo123 | sudo -S -p '' nmcli con mod netplan-eth0 ipv4.method manual ipv4.addresses 192.168.1.91/24 ipv4.gateway 192.168.1.1 ipv4.dns 192.168.1.1 2>/dev/null || (nmcli -t -f NAME con show | head -1); echo coramo123 | sudo -S -p '' nmcli con up netplan-eth0 >/dev/null 2>&1; hostname -I"
ssh coramo@192.168.1.91 "echo coramo123 | sudo -S -p '' apt-get update -qq && echo coramo123 | sudo -S -p '' apt-get install -y -qq libcamera-tools chrony && cam -l"
```
Expected: `192.168.1.91` y `cam -l` listando **2** cámaras (`/base/axi/pcie@.../rp1/i2c@88000/imx…@1a` y `…i2c@80000/…`). Si lista 1 o 0: agregar a `/boot/firmware/config.txt` las líneas de `head/config.txt.snippet` (paso 3) con el modelo real de las cámaras y reiniciar.

- [ ] **Step 3: Archivos de la cabeza en el repo (se escriben en WSL)**

```bash
mkdir -p ~/coramo/head && cat > ~/coramo/head/config.txt.snippet <<'EOT'
# Añadir al final de /boot/firmware/config.txt de la RPi5 si cam -l no ve las dos cámaras.
# Sustituir imx219 por el sensor real (imx219 = Camera Module v2, imx708 = v3, imx477 = HQ).
camera_auto_detect=0
dtoverlay=imx219,cam0
dtoverlay=imx219,cam1
EOT
cat > ~/coramo/head/head_cameras.launch.py <<'EOT'
"""Publica las dos cámaras CSI de la cabeza a 640x480 y 15 FPS con camera_ros."""
from launch import LaunchDescription
from launch_ros.actions import Node

def camara(nombre, indice):
    return Node(package="camera_ros", executable="camera_node", namespace=f"/head/{nombre}", name="camera",
                parameters=[{"camera": indice, "width": 640, "height": 480, "format": "YUYV",
                             "FrameDurationLimits": [66666, 66666]}])  # 15 FPS en microsegundos

def generate_launch_description():
    return LaunchDescription([camara("cam_left", 0), camara("cam_right", 1)])
EOT
cat > ~/coramo/head/head-cameras.service <<'EOT'
[Unit]
Description=Cámaras de la cabeza CORAMO (camera_ros)
After=network-online.target
Wants=network-online.target

[Service]
User=coramo
ExecStart=/bin/bash -lc "source /opt/ros/lyrical/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DISCOVERY_SERVER=192.168.1.90:11811 ROS_DOMAIN_ID=7 && exec ros2 launch /home/coramo/coramo/head/head_cameras.launch.py"
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOT
cat > ~/coramo/head/README.md <<'EOT'
# Cabeza (RPi5)
Ubuntu Server 26.04 + ROS 2 Lyrical base + camera_ros. IP fija 192.168.1.91. Publica /head/cam_left y /head/cam_right (640x480, 15 FPS, JPEG). Servicio: head-cameras.service. Instalación: docs/instalacion/cabeza.md.
EOT
cd ~/coramo && git add head && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -q -m "head: launch de cámaras, servicio y snippet de config" && echo ok
```

- [ ] **Step 4: ROS 2 en la cabeza y sincronía de reloj**

```bash
scp -r ~/coramo/head coramo@192.168.1.91:~/coramo/ 2>/dev/null || (ssh coramo@192.168.1.91 'mkdir -p ~/coramo' && scp -r ~/coramo/head coramo@192.168.1.91:~/coramo/)
ssh coramo@192.168.1.91 "echo coramo123 | sudo -S -p '' apt-get install -y -qq software-properties-common curl && echo coramo123 | sudo -S -p '' add-apt-repository -y universe >/dev/null && V=\$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F'\"' '{print \$4}') && curl -sL -o /tmp/ros2-apt-source.deb \"https://github.com/ros-infrastructure/ros-apt-source/releases/download/\${V}/ros2-apt-source_\${V}.\$(. /etc/os-release && echo \$VERSION_CODENAME)_all.deb\" && echo coramo123 | sudo -S -p '' dpkg -i /tmp/ros2-apt-source.deb && echo coramo123 | sudo -S -p '' apt-get update -qq && echo coramo123 | sudo -S -p '' apt-get install -y -qq ros-lyrical-ros-base ros-lyrical-rmw-fastrtps-cpp ros-lyrical-camera-ros ros-lyrical-compressed-image-transport && printf 'server 192.168.1.90 iburst prefer\n' | sudo -S -p '' tee -a /etc/chrony/chrony.conf >/dev/null <<< coramo123 && echo coramo123 | sudo -S -p '' systemctl restart chrony && sleep 5 && chronyc tracking | grep -E 'Reference ID|System time'"
```
Expected: `Reference ID : C0A8015A (192.168.1.90)` y `System time : 0.00x seconds` (en el Xeon, `chrony` debe permitir clientes: `allow 192.168.1.0/24` en `/etc/chrony/chrony.conf` y reinicio; instalar `chrony` en el Xeon si falta).

- [ ] **Step 5: Servicio de cámaras y prueba de 10 minutos desde el Xeon**

```bash
ssh coramo@192.168.1.91 "echo coramo123 | sudo -S -p '' cp ~/coramo/head/head-cameras.service /etc/systemd/system/ && echo coramo123 | sudo -S -p '' systemctl daemon-reload && echo coramo123 | sudo -S -p '' systemctl enable --now head-cameras && sleep 8 && systemctl is-active head-cameras"
ssh coramo "bash -lc 'ros2 topic list | grep head; timeout 600 ros2 topic hz /head/cam_left/image_raw/compressed 2>&1 | tail -3'"
```
Expected: `active`; la lista muestra `/head/cam_left/image_raw/compressed` y `/head/cam_right/...`; tras 10 min, `average rate: 15.0xx` con `min` no menor de 12 Hz y sin cortes. Guardar la salida en `docs/mediciones/` en la Task 12.

- [ ] **Step 6: Documentar y commit**

```bash
cat > ~/coramo/docs/instalacion/cabeza.md <<'EOT'
# Instalación de la cabeza (RPi5)
- Imagen: Ubuntu Server 26.04 LTS 64-bit (Raspberry Pi Imager), hostname cabeza, usuario coramo, SSH por llave.
- Hardware retirado: X1011, RX 580 y kernel Coreforge de v1. Cámaras CSI x2 (modelo: anotar).
- Red: eth0 fija 192.168.1.91/24, gw 192.168.1.1. Reserva DHCP en el router.
- Cámaras: `cam -l` lista 2. Overlays extra en config.txt: (sí/no).
- ROS 2 Lyrical base + camera_ros + compressed-image-transport. Env: RMW fastrtps, ROS_DISCOVERY_SERVER=192.168.1.90:11811, ROS_DOMAIN_ID=7.
- chrony contra 192.168.1.90.
- Servicio head-cameras.service → /head/cam_left y /head/cam_right a 640x480, 15 FPS.
EOT
cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -q -m "docs(cabeza): instalación de la RPi5 como nodo cabeza" && echo ok
```

---

### Task 12: Bitácora de mediciones y tabla de decisión de backends

**Files:**
- Create: `docs/mediciones/2026-09-XX-hito0.md`
- Modify: `docs/superpowers/specs/2026-09-18-coramo-v2-arquitectura-design.md` (sección 5.6: columna "Elegido")

**Interfaces:**
- Produces: la tabla que fija `xeon.yaml` en el subproyecto A.

- [ ] **Step 1: Volver a correr todos los benchmarks seguidos y guardar la salida**

```bash
ssh coramo 'source ~/.config/coramo/env; F=~/coramo/docs/mediciones/$(date +%F)-hito0.md; { echo "# Hito 0: mediciones ($(date +%F))"; echo; echo "Hardware: Xeon E5-2697 v2, RTX 4070 SUPER, Ubuntu 26.04, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader). Red: cable RTL8153. 30 órdenes sintéticas (tools/bench/ordenes.txt)."; echo; echo "## Resultados"; echo; echo "\`\`\`"; ~/venvs/tts/bin/python ~/coramo/tools/bench/bench_tts.py; ~/venvs/stt/bin/python ~/coramo/tools/bench/bench_stt.py | head -1; ~/venvs/bench/bin/python ~/coramo/tools/bench/bench_llm_local.py | head -1; ~/venvs/bench/bin/python ~/coramo/tools/bench/bench_nube.py; ~/venvs/vision/bin/python ~/coramo/tools/bench/bench_vision.py; echo "\`\`\`"; } | tee $F'
```
Expected: el archivo con las 8 líneas de resultado (TTS local, STT local, LLM local, STT nube, TTS nube, Claude Haiku, Claude Sonnet, YOLO).

- [ ] **Step 2: Escribir la tabla de decisión** (a mano, con los números del paso 1) al final del mismo archivo:

```markdown
## Decisión de backends

| Etapa | Local p50/p95 | Nube p50/p95 | Acierto/WER | Elegido para xeon.yaml | Razón |
|---|---|---|---|---|---|
| STT | | | | | |
| LLM | | Haiku: / Sonnet: | | | |
| TTS | | | | | |
| Visión | | (no aplica) | | local | continuo |

Regla del spec §5.6: gana el de menor latencia que cumpla acierto ≥ 90 % (LLM) o WER ≤ 10 % (STT); a igual latencia (±0,1 s), gana el local por costo cero. Cámaras: `ros2 topic hz` 10 min: (pegar las 3 líneas).
```

- [ ] **Step 3: Actualizar el spec con lo elegido** (editar la tabla de §5.6 en `docs/superpowers/specs/...`: añadir a cada fila el texto "**Elegido (fecha): local|nube**").

- [ ] **Step 4: Commit y sincronizar el clon de WSL**

```bash
ssh coramo 'cd ~/coramo && git add docs && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -m "docs(mediciones): hito 0, latencias por backend y decisión"'
cd ~/coramo && git fetch coramo:~/coramo v2-planificacion 2>/dev/null || git remote add xeon coramo:~/coramo && git fetch xeon v2-planificacion && git merge --ff-only xeon/v2-planificacion && git log --oneline -3
```
Expected: los commits del Xeon aparecen en el clon de WSL.

---

### Task 13 (con permiso explícito de Felipe): gobernanza del repo en GitHub

**Files:**
- GitHub: rama `v1-rpi5`, tag `v1.0`, `main` reiniciado; `docs/legado/` con los `docs/01..06` de v1.

Esta tarea **hace push**. No se ejecuta sin que Felipe lo diga en el momento.

- [ ] **Step 1: Conservar v1**

```bash
cd ~/coramo && git branch v1-rpi5 e7173ad && git tag -a v1.0 e7173ad -m "CORAMO v1: RPi5 + 2x RX 580, pipeline de voz a mano robótica" && git push origin v1-rpi5 v1.0
```

- [ ] **Step 2: Nuevo main con historia limpia**

```bash
cd ~/coramo && git checkout --orphan main-v2 && git rm -rq . && git checkout v2-planificacion -- docs/superpowers head tools && git checkout v1-rpi5 -- docs/01-red.md docs/02-alimentacion.md docs/03-gpu.md docs/04-video.md docs/05-whisper.md docs/06-asistente-voz.md && mkdir -p docs/legado && git mv docs/0*.md docs/legado/ && printf '# CORAMO v2\n\nRobot humanoide COlaborativo Reprogramable Autónomo MOdular. Rediseño 2026.\n\n- Diseño: docs/superpowers/specs/\n- Planes: docs/superpowers/plans/\n- Instalación: docs/instalacion/\n- Mediciones: docs/mediciones/\n- v1 (2026, RPi5 + RX 580): rama v1-rpi5, tag v1.0, docs/legado/\n' > README.md && git add -A && git -c user.name="Felipe Ballesteros" -c user.email="felipe1024@gmail.com" commit -qm "CORAMO v2: inicio limpio (spec, plan del hito 0, docs de v1 en legado)" && git branch -M main-v2 main && git push --force-with-lease origin main
```
Expected: GitHub muestra `main` con README nuevo, `docs/legado/`, y la rama `v1-rpi5` intacta.

---

## Self-review (hecho al escribir el plan)

- **Cobertura del spec, hito 0:** Ubuntu 26.04 (T0), driver NVIDIA y CUDA por pip (T3), RX 580 como pantalla (T1), sin suspensión ni ahorro de energía en red (T1b), red por cable RTL8153 (T2), fuente (T4), ROS 2 Lyrical + Discovery Server (T10), servidores de modelo en uv 3.12 (T5–T7 crean los venvs; los servidores HTTP propios de STT/TTS son del subproyecto A), RPi5 cabeza con 26.04 + Lyrical (T11), tabla de latencia por backend p50/p95 de 30 peticiones (T5–T8, T12), FPS del detector (T9), cámaras visibles desde el Xeon con `hz` de 10 min (T11), tabla de decisión (T12).
- **Fuera del hito 0, a propósito:** grabar voces reales (spec §6.4, subproyecto A), nodos ROS del cerebro, protocolo del Pico.
- **Consistencia:** `comun.cargar_ordenes/medir/imprimir` se definen en T5 y se usan en T6, T7, T8; `tools_coramo.json` se define en T7 y se usa en T8; IPs 192.168.1.90/.91 y `ROS_DOMAIN_ID=7` iguales en T2, T10, T11.
- **Riesgos abiertos que el plan no puede resolver por adelantado:** nombre exacto del paquete CUDA en el archivo de 26.04 (T3 trae alternativa), modelo de las cámaras CSI (T11 paso 2 y snippet), existencia de `--chat-template-kwargs` en la versión de llama.cpp clonada (T7 trae alternativa), y si `claude-sonnet-5` acepta `thinking disabled` con el effort por defecto (T8 trae alternativa).
