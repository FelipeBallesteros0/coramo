# Instalación de la cabeza (RPi5)

Nodo de visión de CORAMO v2. Estado al 2026-09-20: arranca desde SSD NVMe, con las dos cámaras operativas y red por cable y WiFi. Falta ROS 2 y el servicio de cámaras.

## Hardware

| Elemento | Detalle |
|---|---|
| Placa | Raspberry Pi 5 Model B Rev 1.0 |
| Shield M.2 | Sin conmutador PCIe: el SSD cuelga directo del puente del BCM2712 (`0001:01:00.0`). No aplica la limitación de arranque NVMe detrás de un ASM1184e. |
| SSD | `nvme0n1`, 238,5 GB, MAXIO MAP1202 (NXM-256 2242, DRAM-less) |
| Cámaras | 2× OV5647 (Camera Module v1, 5 MP), una por puerto CSI |
| Red | `eth0` integrada (cable) y `wlan0` (WiFi 5 GHz). El adaptador USB Realtek de v1 también está presente. |

Retirado de v1: RX 580, multiplexor X1011 y kernel Coreforge.

## Sistema

- **Ubuntu Server 26.04.1 LTS**, kernel 7.0.0-1017-raspi. Usuario `coramo`, hostname `cabeza`. Acceso por llave: `ssh cabeza` (alias en el `~/.ssh/config` de WSL).
- **Arranca desde el SSD.** Migración hecha con `head/migrar_a_ssd.sh`. Arranque completo en **6,3 s** (448 ms kernel + 1,4 s initrd + 4,4 s userspace).

| Unidad | Etiquetas | Raíz |
|---|---|---|
| SSD (en uso) | `SSDBOOT` / `writable-ssd` | `root=PARTUUID=d6b594f8-02` |
| microSD (retirada, guardada como rescate) | `system-boot` / `writable` | `root=LABEL=writable` |

- EEPROM: `BOOT_ORDER=0xf416` (NVMe, luego microSD, luego USB) y `PCIE_PROBE=1`.
- Lectura medida: **SSD 434 MB/s contra microSD 82,6 MB/s**.

## Trampas encontradas (2026-09-20)

1. **El usuario no queda en el grupo `video`.** cloud-init crea al usuario sin él, así que libcamera falla con `Permission denied` en `/dev/media*` y `cam -l` no lista nada. Arreglo: `sudo usermod -aG video,render coramo`. Hace falta también para `head-cameras.service`, que corre como ese usuario.

2. **Cambiar la etiqueta de la partición de arranque rompe cloud-init.** `/etc/cloud/cloud.cfg.d/99-fake-cloud.cfg` busca el seed por `fs_label: system-boot`. Al migrar al SSD la partición quedó como `SSDBOOT`, cloud-init no encontró el seed, cayó a `DataSourceNone` (`instance-id: iid-datasource-none`) y **regeneró `/etc/netplan/50-cloud-init.yaml` sin la sección `wifis`**: declaró `wlan0` como si fuera Ethernet, sin SSID ni contraseña. Mientras la microSD estuvo puesta no se notó, porque cloud-init leía el seed de ella.
   - Arreglo inmediato: `sudo cp /boot/firmware/network-config /etc/netplan/50-cloud-init.yaml` (ese archivo ya viene en formato netplan) y `sudo netplan apply`.
   - Arreglo permanente: apuntar cloud-init a la etiqueta real, `fs_label: SSDBOOT`. Verificado: tras reiniciar, `cloud-init query` devuelve `instance-id: rpi-imager-...` y el netplan conserva `wifis`.
   - Copia de seguridad del netplan bueno en `/root/50-cloud-init.yaml.bak`.

3. **Sin `iw` no llega el país al chip WiFi** y la banda de 5 GHz queda bloqueada (`brcmfmac: set chanspec 0x... fail, reason -52` en los canales 38-46 y 149-165). Misma trampa que en la Raspberry del 4WD. Arreglo: `apt install iw wireless-regdb` y `iw reg set CL`; netplan ya trae `regulatory-domain: "CL"`, que necesita `iw` para aplicarse.

4. **Vigilar tras cada actualización de kernel:** `linux-firmware-raspi` gestiona `/boot/firmware/current/` con esquema A/B (`tryboot_a_b=1`). Si regenerara `current/cmdline.txt` con `root=LABEL=writable`, no encontraría la raíz (el SSD se llama `writable-ssd`). Comprobar con `findmnt -n -o SOURCE /`.

## Cámaras

Las dos detectadas y capturando a 29 fps. `camera_auto_detect=1` basta; no hacen falta overlays (`head/config.txt.snippet` queda solo por si se cambian los sensores). Usar los **IDs** en `camera_ros`, no el índice, porque el orden de enumeración puede cambiar entre arranques:

- `/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36`
- `/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36`

## Red

| Interfaz | Dirección | Uso |
|---|---|---|
| `wlan0` | 192.168.1.104 (DHCP, conviene reservarla en el router) | **Enlace real con el Xeon** y salida a internet |
| `eth0` | 192.168.50.2/24 fija (`/etc/netplan/60-enlace-directo.yaml`) | Enlace directo con el Xeon. **Configurado pero inservible hoy**, ver abajo |

**Ahorro de energía del WiFi apagado.** Estaba encendido y provocaba picos de latencia: el ping desde el Xeon daba 60 ms de media con máximos de 163 ms. Apagado baja a **8,9 ms de media, máximo 11,3 ms**. Como la Pi usa netplan con systemd-networkd (no NetworkManager), se hizo permanente con el servicio `wifi-powersave-off.service`, que ejecuta `iw dev wlan0 set power_save off` al arrancar. Misma trampa que en la Raspberry del 4WD y en el Xeon.

### Enlace directo por cable: no funciona con el hardware actual

Se configuró el enlace punto a punto que pide el spec (Xeon 192.168.50.1 en su adaptador USB Realtek RTL8153, cabeza 192.168.50.2 en `eth0`, sin puerta de enlace, rutas por defecto intactas en el WiFi de ambas). **La configuración de software es correcta en los dos extremos y queda puesta**, pero el enlace no transporta datos:

- El enlace negocia 1000 Mb/s y ambos extremos dicen "link detected", pero el adaptador del Xeon marca **0 paquetes recibidos** siempre.
- La captura en la cabeza muestra que sí llegan las peticiones del Xeon y que la cabeza responde; **las respuestas nunca llegan al Xeon**. El fallo es en un solo sentido.
- `dmesg` del Xeon registra **13 caídas y subidas de portadora** en diez minutos.
- Descartado por software: no es velocidad (falla igual forzando 100 Mb/s), ni EEE (desactivado), ni autosuspend USB (`power/control=on`), ni cortafuegos (ambos inactivos), ni el driver (recargado y re-enumerado; funcionó unos segundos y volvió a fallar).

Conclusión: **cable o adaptador defectuoso**. A probar, en orden: otro cable Ethernet, otro puerto USB del Xeon, otro adaptador USB. La red integrada de la placa P9X79 LE no sirve como alternativa: no aparece como dispositivo en el sistema, aunque el descriptor DMI diga "Onboard LAN: Enabled".

Mientras tanto, **el WiFi cumple de sobra**: el video comprimido de las dos cámaras son unos 1,2 MB/s y ambas máquinas están en 5 GHz con menos de 9 ms de latencia.

## Pendiente

- Reservar 192.168.1.104 para la cabeza en el router (hoy es DHCP).
- Probar otro cable para el enlace directo.
- ROS 2 Lyrical Luth, `camera_ros`, `chrony` contra el Xeon y el servicio `head-cameras.service`.
