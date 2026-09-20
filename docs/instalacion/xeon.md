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
