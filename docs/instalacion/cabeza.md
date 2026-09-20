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

### Enlace directo por cable: entrena pero pierde el 94 % del tráfico

Se configuró el enlace punto a punto que pide el spec (Xeon 192.168.50.1 en su adaptador USB Realtek RTL8153, cabeza 192.168.50.2 en `eth0`, sin puerta de enlace, rutas por defecto intactas en el WiFi de ambas). **La configuración de software es correcta**, pero el enlace no entrega los datos:

| Medida | Valor |
|---|---|
| Enviado por la cabeza (`eth0` TX) | 1,37 GB |
| Recibido por el Xeon (adaptador USB RX) | 89 MB |
| Entrega efectiva | ~6 % |

- El enlace negocia **1000 Mb/s full duplex** y la portadora es estable; los contadores de error de ambas interfaces marcan **cero**.
- El Xeon responde con **`ICMP ip reassembly time exceeded`** de forma continua: le llegan fragmentos sueltos de los mensajes de imagen y nunca completa el datagrama.
- La resolución ARP termina en `INCOMPLETE`, así que hasta un `ping` deja de salir.
- **Efecto sobre las cámaras:** Fast DDS anuncia todas las interfaces, así que eligió este camino para el video. Por eso el Xeon veía **1,2 Hz con cortes de 33 s** mientras la cabeza publicaba 15,005 Hz perfectos.

Descartado por software: velocidad (falla igual a 100 Mb/s), EEE, autosuspend USB, cortafuegos (ambos vacíos), driver (`r8152` recargado y dispositivo re-enumerado) y modo del adaptador (probado también CDC-ECM).

**Queda desactivado en ambos extremos** hasta tener hardware sano: en el Xeon la conexión `enlace-cabeza` con `autoconnect no`, en la cabeza el netplan movido a `/root/60-enlace-directo.yaml.deshabilitado` y `eth0` abajo. Con solo WiFi las cámaras van a 15 Hz estables.

**No está determinado si falla el cable o el adaptador.** Prueba pendiente, una sola acción: conectar el adaptador USB del Xeon al router con ese mismo cable. Si toma dirección por DHCP, ambos sirven y el problema es específico del enlace punto a punto; si no, se cambia primero el cable y luego el adaptador. La red integrada de la placa P9X79 LE no es alternativa: su puerto PCIe reporta ancho de enlace cero, el chip está muerto (ver `docs/instalacion/xeon.md`).

## ROS 2 y cámaras

- **ROS 2 Lyrical Luth** desde `ros2-apt-source` (207 paquetes): `ros-lyrical-ros-base`, `rmw-fastrtps-cpp`, `camera-ros`, `compressed-image-transport`.
- Entorno en `/etc/profile.d/coramo-ros.sh`: `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DISCOVERY_SERVER=192.168.1.103:11811` (el Xeon, por WiFi), `ROS_DOMAIN_ID=7`.
- `chrony` con el Xeon como servidor preferido; el Xeon acepta clientes con `allow 192.168.1.0/24`.
- Servicio **`head-cameras.service`**, arranca solo y se reinicia si falla. Publica:

| Tema | Contenido |
|---|---|
| `/head/cam_left/image_raw` y `/compressed` | cámara izquierda, 640×480 |
| `/head/cam_right/image_raw` y `/compressed` | cámara derecha, 640×480 |
| `/head/cam_left/camera_info`, `/head/cam_right/camera_info` | calibración (aún sin calibrar) |

**Medido desde el Xeon:** 15,07 Hz de media en `/head/cam_left/image_raw/compressed`, con mínimo 0,049 s y máximo 0,084 s entre cuadros, desviación 0,008 s. Es el objetivo de 15 FPS del spec, sobre WiFi.

### Trampa: camera_ros trae su propia libcamera y no ve el hardware

`ros-lyrical-camera-ros` depende de `ros-lyrical-libcamera` **0.7.2**, que convive con la del sistema (`libcamera0.7` **0.7.0** de Ubuntu). El nodo enlaza contra la de ROS y falla:

```
RPI pisp.cpp: Unable to acquire a CFE instance
terminate called ... what(): no cameras available
```

El pipeline `rpi/pisp` sí está registrado en la de ROS, pero no reconoce el frontal de cámara de este kernel. La del sistema sí (`cam -l` lista las dos). Como **ambas comparten el soname `libcamera.so.0.7`**, basta anteponer la ruta del sistema:

```
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

Eso va en `ExecStart` de `head-cameras.service`. Sin ello el servicio arranca y muere en bucle.

Nota menor: `camera_calibration_parsers` avisa que no encuentra el archivo de calibración. Es esperable hasta que se calibren las cámaras; no impide publicar.

## Pendiente

- Reservar 192.168.1.104 para la cabeza en el router (hoy es DHCP).
- Determinar si el enlace directo falla por el cable o por el adaptador (prueba contra el router).
- Calibrar las dos cámaras (`camera_calibration`) para llenar `camera_info`.
- Prueba de 10 minutos de `ros2 topic hz` registrada en `docs/mediciones/`.
