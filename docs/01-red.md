# 01 - Conectividad de red

## Problema

Ubuntu 24.04 en Raspberry Pi 5 presenta cortes intermitentes en el WiFi integrado y el ethernet integrado.

## Hardware de red disponible

| Interfaz | Tipo | Driver |
|---|---|---|
| `eth0` | Ethernet integrado (BCM54213PE, PCIe) | `macb` |
| `wlan0` | WiFi integrado (CYW43455, SDIO) | `brcmfmac` |
| `wlx90de80052ea8` | Antena WiFi USB (MediaTek MT7921U) | `mt7921u` |
| `enxf8ce21123f7b` | Hub USB RJ45 (Realtek RTL8151) | `r8152` |

## Causas identificadas

1. **WiFi integrado (wlan0)**: dominio regulatorio incorrecto (`country 99: DFS-UNSET`), fallo al usar canales 5GHz
2. **USB autosuspend**: dispositivos USB se desconectan solos por gestión de energía
3. **Power management WiFi**: el chip entra en modo ahorro y pierde conexión

## Soluciones aplicadas

### 1. Dominio regulatorio

```bash
echo 'REGDOMAIN=CL' | sudo tee /etc/default/crda
sudo tee /etc/modprobe.d/cfg80211.conf << 'EOF'
options cfg80211 ieee80211_regdom=CL
EOF
```

### 2. Desactivar USB autosuspend

```bash
sudo tee /etc/udev/rules.d/70-usb-power.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0e8d", ATTRS{idProduct}=="7961", ATTR{power/autosuspend}="-1"
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="8151", ATTR{power/autosuspend}="-1"
ACTION=="add", SUBSYSTEM=="usb", TEST=="power/autosuspend", ATTR{power/control}="on"
EOF
```

### 3. Desactivar power management WiFi

```bash
sudo tee /etc/udev/rules.d/70-wifi-power.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="net", KERNEL=="wlan0", RUN+="/sbin/iwconfig wlan0 power off"
ACTION=="add", SUBSYSTEM=="net", KERNEL=="wlx*", RUN+="/sbin/iwconfig %k power off"
EOF
```

### 4. Activar RTL8151 (hub USB RJ45)

El RTL8151 arranca en modo mass storage. Se activa con:

```bash
sudo tee /etc/udev/rules.d/71-rtl8151-modeswitch.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="8151", RUN+="/usr/sbin/usb_modeswitch --reset-usb -v 0x0bda -p 0x8151"
EOF
```

### 5. Routing con failover automático

Métricas configuradas para failover automático:

| Interfaz | Métrica | Prioridad |
|---|---|---|
| `eth0` | 100 | Principal |
| `wlx90de80052ea8` | 200 | Backup WiFi USB |
| `enxf8ce21123f7b` | 300 | Backup RJ45 USB |

```bash
nmcli connection modify netplan-eth0 ipv4.route-metric 100
nmcli connection modify movistar5GHZ_442777 ipv4.route-metric 200
```

Script dispatcher para que la ruta WiFi USB persista en cada arranque:

```bash
sudo tee /etc/NetworkManager/dispatcher.d/10-wifi-route << 'EOF'
#!/bin/bash
IFACE="$1"
ACTION="$2"
if [ "$IFACE" = "wlx90de80052ea8" ] && [ "$ACTION" = "up" ]; then
    sleep 2
    ip route add 192.168.1.0/24 dev wlx90de80052ea8 metric 200 src 192.168.1.103 2>/dev/null
    ip route add default via 192.168.1.1 dev wlx90de80052ea8 metric 200 2>/dev/null
fi
EOF
sudo chmod +x /etc/NetworkManager/dispatcher.d/10-wifi-route
```

### 6. Firmwares .zst (kernel 6.6.x)

El kernel Coreforge 6.6.70 no descomprime `.zst` automáticamente. Descomprimir todos los firmwares:

```bash
for dir in /lib/firmware /lib/firmware/brcm /lib/firmware/amdgpu /lib/firmware/mediatek /lib/firmware/rtl_nic; do
    for f in "$dir"/*.zst 2>/dev/null; do
        sudo zstd -d "$f" -o "${f%.zst}" --force -q
    done
done
```

## Verificación

```bash
nmcli device status
ip route show
ping -c 3 8.8.8.8
ping -c 3 -I wlx90de80052ea8 8.8.8.8
```
