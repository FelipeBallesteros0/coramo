# 04 - Salida de video por GPU

## Contexto

Con amdgpu cargado, la RX 580 expone sus salidas de video a GNOME/Wayland. El RPi5 tiene también su propia salida HDMI integrada (VC4), ambas coexisten.

## Conectores de la RX 580

```
card1: DP-1, DP-2, DP-3, DVI-D-1, HDMI-A-3
card3: DP-4, DP-5, DP-6, DVI-D-2, HDMI-A-4
```

## Configurar resolución (GNOME/Wayland)

Editar `~/.config/monitors.xml`:

```xml
<monitors version="2">
  <configuration>
    <logicalmonitor>
      <x>0</x>
      <y>0</y>
      <scale>1</scale>
      <primary>yes</primary>
      <monitor>
        <monitorspec>
          <connector>HDMI-3</connector>
          <vendor>SAM</vendor>
          <product>SAMSUNG</product>
          <serial>0x01000e00</serial>
        </monitorspec>
        <mode>
          <width>1920</width>
          <height>1080</height>
          <rate>60.000</rate>
        </mode>
      </monitor>
    </logicalmonitor>
  </configuration>
</monitors>
```

Aplica al reiniciar sesión GNOME.

## Modos disponibles (HDMI-A-3)

```
3840x2160 @ 60Hz, 30Hz
2560x1440 @ 59.95Hz
1920x1080 @ 60Hz, 59.94Hz, 50Hz
1920x1200 @ 30Hz
1680x1050 @ 59.88Hz
...
```

## Render offload con DRI_PRIME

Para usar la RX 580 en apps específicas manteniendo el escritorio en VC4:

```bash
# Usar RX 580 #1
DRI_PRIME=pci-0000_03_00_0 glxinfo | grep "OpenGL renderer"

# Usar RX 580 #2
DRI_PRIME=pci-0000_05_00_0 glxinfo | grep "OpenGL renderer"
```

## Verificación

```bash
# Ver salidas conectadas
ls /sys/class/drm/card1-*
cat /sys/class/drm/card1-HDMI-A-3/status

# Ver modos disponibles
modetest -M amdgpu -c 2>/dev/null | grep -A30 "HDMI-A-3"
```
