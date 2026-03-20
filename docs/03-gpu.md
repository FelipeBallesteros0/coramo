# 03 - GPU AMD RX 580 en Raspberry Pi 5

## Contexto

El kernel oficial de Ubuntu 24.04 (`6.8.x-raspi`) no incluye soporte para `amdgpu` en aarch64. Se requiere compilar el kernel **Coreforge** (`rpi-6.6.y-gpu`) que contiene los parches necesarios.

## Hardware

- **GPU**: AMD Radeon RX 580 2048SP (Polaris10, 8GB GDDR5)
- **Expansión**: Suptronics X1011 con ASMedia ASM1184e (PCIe x1 Gen2, 4 puertos)
- **Alimentación GPU**: conector PCIe 8-pin obligatorio

## Paso 1 — Habilitar PCIe Gen3

Añadir a `/boot/firmware/config.txt`:

```ini
dtparam=pciex1
dtparam=pciex1_gen=3
```

## Paso 2 — Dependencias de compilación

```bash
sudo apt install -y git bc bison flex libssl-dev libncurses-dev \
    make gcc build-essential libelf-dev
```

## Paso 3 — Compilar kernel Coreforge

```bash
git clone --depth=1 --branch rpi-6.6.y-gpu https://github.com/Coreforge/linux.git
cd linux
export KERNEL=kernel_2712
make bcm2712_defconfig

# Activar opciones necesarias
scripts/config --enable CONFIG_DRM_AMDGPU
scripts/config --enable CONFIG_DRM_AMDGPU_SI
scripts/config --enable CONFIG_DRM_AMDGPU_CIK
scripts/config --enable CONFIG_ARM64_ALIGNMENT_FIXUPS
scripts/config --enable CONFIG_COMPAT_ALIGNMENT_FIXUPS
make olddefconfig

# Compilar (1-2 horas en RPi5)
make -j4 Image.gz modules dtbs
sudo make -j4 modules_install
```

## Paso 4 — Instalar kernel

```bash
export KERNEL=kernel_2712
sudo cp arch/arm64/boot/Image.gz /boot/firmware/$KERNEL.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/

# Apuntar config.txt al nuevo kernel
sudo sed -i "s/^kernel=vmlinuz/kernel=kernel_2712.img/" /boot/firmware/config.txt
```

## Paso 5 — Firmware AMD

El kernel 6.6.70 no descomprime `.zst` automáticamente:

```bash
# Descomprimir firmwares Polaris (RX 580)
cd /lib/firmware/amdgpu
for f in polaris*.zst; do sudo zstd -d "$f" -o "${f%.zst}" --force -q; done
```

## Paso 6 — Fix de memory alignment

Crítico para aarch64 con amdgpu:

```bash
cat << 'EOF' > /tmp/memcpy_unaligned.c
#include <string.h>
void *memcpy(void *dest, const void *src, size_t n) {
    char *d = dest; const char *s = src;
    while (n--) *d++ = *s++;
    return dest;
}
EOF
gcc -shared -fPIC -O2 -o /tmp/memcpy.so /tmp/memcpy_unaligned.c
sudo mv /tmp/memcpy.so /usr/local/lib/memcpy.so
echo "/usr/local/lib/memcpy.so" | sudo tee /etc/ld.so.preload
```

> ⚠️ Este preload causa segfault con `sudo modprobe` en caliente. El módulo carga correctamente en el boot.

## Verificación

```bash
uname -r
# 6.6.70-v8-16k+

lspci | grep AMD
# 0000:03:00.0 VGA compatible controller: AMD Polaris 20 XL [Radeon RX 580 2048SP]
# 0000:05:00.0 VGA compatible controller: AMD Polaris 20 XL [Radeon RX 580 2048SP]

lsmod | grep amdgpu
dmesg | grep "Initialized amdgpu"
ls /dev/dri/
```

## Mapa de dispositivos DRM

| Device | GPU |
|---|---|
| `card0` | VC4 integrada RPi5 |
| `card1` | RX 580 #1 (`03:00.0`) → `renderD129` |
| `card2` | V3D integrada RPi5 → `renderD128` |
| `card3` | RX 580 #2 (`05:00.0`) → `renderD130` |

## Notas conocidas

- `fp16: 0` en Vulkan RADV — la RX 580 corre en fp32 en este driver. Afecta velocidad de inferencia en modelos cuantizados.
- `Cannot find any crtc or sizes` — normal si no hay monitor conectado a esa GPU.
- El módulo amdgpu solo carga correctamente desde boot, no con `modprobe` en caliente (problema de IRQ mapping con el ASM1184e).
