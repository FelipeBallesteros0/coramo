# Instalación del Xeon (cerebro CORAMO v2)

Inventario base (2026-09-20): ASUS P9X79 LE, Xeon E5-2697 v2, 8x8 GB DDR3-1600, RTX 4070 SUPER (01:00.0, PCIe 3.0 x16), RX 580 (02:00.0), SSD SATA 240 GB, Ubuntu 26.04.1, kernel 7.0, nvidia-driver-595-open.

Cada sección de abajo la agrega la tarea del plan que la ejecuta.

## Energía (tarea 1b, 2026-09-20)
- systemd: sleep.target, suspend.target, hibernate.target e hybrid-sleep.target enmascarados (persisten tras reinicio).
- GNOME del usuario coramo y de GDM: sleep-inactive-ac-type = nothing, idle-delay 0.
- /etc/udev/rules.d/70-usb-power.rules: MediaTek 0e8d:7961 y Realtek 0bda:8153 con power/control=on (verificado tras reinicio).
- WiFi: /etc/NetworkManager/conf.d/wifi-powersave-off.conf (wifi.powersave = 2). Ping por WiFi: de 121 ms a 3 ms.
- Conexión WiFi movistar5GHZ_442777 sin restricción de usuario y con autoconnect: sube al arrancar sin login.
- loginctl enable-linger coramo: PipeWire y WirePlumber corren desde el arranque sin sesión gráfica.
- Incidente: un archivo de conf mal escrito (contenía la contraseña por un here-string que pisó la tubería) dejó NetworkManager sin arrancar tras el primer reinicio; corregido, patrón seguro anotado en el plan.

## Audio (tarea 1c, 2026-09-20)
- Micrófono analógico (segundo probado) en el jack rosado trasero (Rear Mic) y parlante en la salida verde de la ALC892 integrada. Sin audio USB.
- PipeWire: fuente alsa_input.pci-0000_00_1b.0.analog-stereo a 0,35 (= Rear Mic Boost 0 dB, Capture 100 %); sumidero alsa_output.pci-0000_00_1b.0.analog-stereo a 0,60. Persisten tras reinicio.
- Nunca subir Rear Mic Boost: con +10 dB el fondo pasa de -49 a -15 dBFS y satura. Ajustar solo con wpctl (WirePlumber pisa amixer).
- Voz a 30 cm: -22 dBFS sobre fondo -42 dBFS (SNR 19-20 dB), pico -6,6 dBFS, sin recortes. Micrófono 1 descartado (SNR 14 dB).
- Transitorio de ~2 s al abrir la captura; audio_check.sh lo descarta. Prueba acústica parlante->mic: tono 440 Hz > 40 dB sobre el fondo.

## Pantalla (tarea 1, 2026-09-20)
- Monitor en el HDMI de la RX 580 (card0, vendor 0x1002). La 4070 sin monitor.
- Mutter elegía la 4070 como GPU primaria (270 MiB de gnome-shell) aunque el monitor estuviera en la RX 580. Arreglo por software, sin tocar la BIOS: /etc/udev/rules.d/61-mutter-primary-gpu.rules etiqueta la tarjeta 0000:02:00.0 con mutter-device-preferred-primary.
- Tras reiniciar: la 4070 queda con 13 MiB usados; gnome-shell solo la abre con 3 MiB (sin renderizar). Criterio cumplido.
- La sesión gráfica de coramo arranca sola al encender.

## Herramientas (tarea 3, 2026-09-20)
- apt: build-essential, cmake 4.2.3, git 2.53, git-lfs, curl, wget, htop, nvtop, espeak-ng 1.52, ffmpeg, libsndfile1, alsa-utils, python3-venv, chrony (activo), nvidia-cuda-toolkit 12.4 (nvcc en /usr/bin; el archivo de 26.04 trae 12.4, suficiente para compilar llama.cpp con CUDA sobre el driver 595).
- uv 0.12.17 en ~/.local/bin (PATH agregado a ~/.bashrc); Python 3.12.14 gestionado por uv. Los venvs viven en ~/venvs/<nombre>.
- ~/venvs/cuda-check: torch 2.14.0+cu130, torch.cuda.is_available() = True, RTX 4070 SUPER.
- Trampas: la imagen de 26.04 no traía curl ni git; el instalador de uv responde 403 a urllib de Python (usar curl).
- Disco tras instalar: 28 GB usados de 218.

## Fuente (tarea 4, 2026-09-20)
- Etiqueta: 750 W (dato de Felipe). Conectores: dos de 8 pines al adaptador de la 4070 y uno a la RX 580.
- Prueba gpu_load.py 120 s en la 4070: potencia clavada en el límite de 220 W, temperatura de 47 a 82 °C, reloj 2,72 a 2,67 GHz (leve throttling térmico al final), PCIe gen 3 x16 bajo carga, 74 TFLOPS fp16.
- Errores de kernel (Xid, reset, PCIe): 0. Sin reinicios. La fuente aguanta la 4070 a tope con la RX 580 y el Xeon activos.
- Nota: la temperatura llega a 82 °C en 2 min; para cargas largas conviene revisar el flujo de aire del gabinete.

## Modelos y entornos (tareas 5 a 9, 2026-09-20)
- ~/venvs/tts: kokoro + soundfile (Python 3.12). TTS Kokoro ef_dora en GPU: p50 0,13 s / p95 0,14 s al primer audio.
- ~/venvs/stt: faster-whisper + nvidia-cublas-cu12 + nvidia-cudnn-cu12 + jiwer; CTranslate2 4.8.2. Whisper large-v3-turbo fp16: p50 0,16 s / p95 0,17 s, WER 8,7 % sobre las 30 órdenes sintéticas.
- ~/llama.cpp compilado con CUDA 12.4 (-DGGML_CUDA=ON, 24 hilos). Modelo ~/modelos/Qwen3-8B-Q5_K_M.gguf (5,85 GB, Qwen/Qwen3-8B-GGUF). Lanzador tools/bench/llama-server.sh (puerto 8080, ctx 8192, KV q8_0, thinking off). 6,1 GiB de VRAM. Benchmark: p50 0,35 s / p95 0,68 s, 29/30 tools correctas.
- ~/venvs/bench: requests, openai, anthropic, soundfile, numpy (clientes de benchmark).
- ~/venvs/vision: ultralytics; YOLO11n a 640x480: 46 FPS, 21,8 ms/cuadro.
- ~/datos/ordenes/NN.wav: 30 órdenes sintéticas (Kokoro, 16 kHz) generadas con tools/bench/gen_ordenes.py.
- Trampa: pkill -f llama-server por SSH mata al propio shell; usar el patrón build/bin/[l]lama-server.

## ROS 2 (tarea 10, 2026-09-20)
- Lyrical Luth desde ros2-apt-source 1.3.0: ros-lyrical-ros-base 0.13.0, ros-dev-tools, rmw-fastrtps-cpp, foxglove-bridge, demo-nodes-cpp (204 paquetes).
- /etc/profile.d/coramo-ros.sh (cargado desde ~/.bashrc): setup.bash, RMW_IMPLEMENTATION=rmw_fastrtps_cpp, ROS_DISCOVERY_SERVER=192.168.1.103:11811, ROS_DOMAIN_ID=7. ros2 doctor: distribution lyrical, middleware rmw_fastrtps_cpp.
- Servicio fastdds-discovery.service (fastdds discovery -i 0 -l 0.0.0.0 -p 11811, usuario coramo, Restart=always). talker/listener por el Discovery Server: OK (6 mensajes en 8 s).
- foxglove_bridge en el puerto 8765 (lanzado a mano; en el subproyecto A pasa a launch/servicio). Conexión desde Windows: ws://192.168.1.103:8765.
- Mientras el Xeon siga por WiFi, reservar 192.168.1.103 en el router: la cabeza y Foxglove apuntan a esa IP.

## Red integrada: el chip está muerto (2026-09-20)

La P9X79 LE trae un **Realtek LAN** integrado y en la BIOS figura como `Realtek LAN Controller: Enabled` (Advanced → Onboard Devices Configuration), pero Linux no lo ve: `lspci` no lista ninguna controladora de red y el driver `r8169` no encuentra hardware.

La causa está en el bus PCIe. El chip cuelga del **puerto raíz 1 del chipset (`00:1c.0`)**, y ese puerto reporta:

```
LnkSta: Speed 2.5GT/s, Width x0
```

`Width x0` significa que **el enlace nunca entrena**: no hay nada al otro lado. El bus 05 está vacío, mientras que los puertos 3, 4 y 5 sí tienen sus dispositivos (dos ASM1042 USB 3.0 y el ASM1061 SATA). No es un ajuste de BIOS ni un driver: el chip de red no responde eléctricamente.

**Conclusión:** el Xeon no tiene ni tendrá puerto Ethernet propio. Para red cableada hace falta un adaptador USB, y el que había (Realtek RTL8153, `0bda:8153`) **también está averiado**: transmite y recibe cero paquetes pese a negociar 1 Gb/s, y el bus USB registra desconexiones (`status -108`). Mientras no haya un adaptador sano, el Xeon va por WiFi, que para esta carga sobra.

