#!/bin/bash
# Copia el sistema de la microSD al SSD NVMe de la cabeza y lo deja arrancable.
# La microSD queda intacta: BOOT_ORDER queda 0xf416 = prueba NVMe, luego microSD, luego USB.
# OJO: BORRA todo el contenido del SSD /dev/nvme0n1 (hoy está vacío, sin tabla de particiones).
set -e
S() { echo coramo123 | sudo -S -p '' "$@"; }
SSD=/dev/nvme0n1

echo "### 0. comprobaciones de seguridad"
[ -b $SSD ] || { echo "no existe $SSD"; exit 1; }
findmnt -n -o SOURCE / | grep -q mmcblk || { echo "la raíz NO está en la microSD; aborto"; exit 1; }
if S fdisk -l $SSD 2>/dev/null | grep -q "^${SSD}p"; then
  echo "el SSD YA tiene particiones; aborto por seguridad. Revísalo a mano."; exit 1
fi
echo "  raíz en $(findmnt -n -o SOURCE /), SSD vacío de $(lsblk -dno SIZE $SSD)"

echo "### 1. particionar el SSD (MBR: 512M FAT32 + resto ext4)"
S wipefs -a $SSD >/dev/null
S parted -s $SSD mklabel msdos
S parted -s $SSD mkpart primary fat32 1MiB 513MiB
S parted -s $SSD set 1 boot on
S parted -s $SSD mkpart primary ext4 513MiB 100%
S partprobe $SSD
sleep 3
S mkfs.vfat -F32 -n SSDBOOT ${SSD}p1 >/dev/null
S mkfs.ext4 -q -F -L writable-ssd ${SSD}p2
lsblk -o NAME,SIZE,FSTYPE,LABEL,PARTUUID $SSD

echo "### 2. copiar la raíz (rsync, solo este sistema de archivos)"
S mkdir -p /mnt/ssd /mnt/ssdboot
S mount ${SSD}p2 /mnt/ssd
S rsync -aHAXx --exclude=/lost+found / /mnt/ssd/
S mkdir -p /mnt/ssd/boot/firmware

echo "### 3. copiar /boot/firmware"
S mount ${SSD}p1 /mnt/ssdboot
S rsync -rltDH --exclude="System Volume Information" /boot/firmware/ /mnt/ssdboot/ || true

echo "### 4. apuntar el SSD a sus propias particiones (por PARTUUID, no por etiqueta)"
P1=$(lsblk -no PARTUUID ${SSD}p1)
P2=$(lsblk -no PARTUUID ${SSD}p2)
echo "  boot PARTUUID=$P1 | raíz PARTUUID=$P2"
printf 'PARTUUID=%s\t/\text4\tdefaults\t0 1\nPARTUUID=%s\t/boot/firmware\tvfat\tdefaults\t0 1\n' "$P2" "$P1" > /tmp/fstab.ssd
S cp /tmp/fstab.ssd /mnt/ssd/etc/fstab
for f in /mnt/ssdboot/cmdline.txt /mnt/ssdboot/current/cmdline.txt; do
  [ -f "$f" ] && S sed -i "s|root=LABEL=writable|root=PARTUUID=$P2|g" "$f"
done
echo "  fstab del SSD:"
cat /mnt/ssd/etc/fstab
echo "  cmdline del SSD:"
grep -o "root=[^ ]*" /mnt/ssdboot/current/cmdline.txt

echo "### 5. desmontar"
S umount /mnt/ssdboot /mnt/ssd

echo "### 6. orden de arranque: NVMe primero, microSD como alternativa"
S rpi-eeprom-config > /tmp/ee.txt
if grep -q '^BOOT_ORDER=' /tmp/ee.txt; then
  sed -i 's/^BOOT_ORDER=.*/BOOT_ORDER=0xf416/' /tmp/ee.txt
else
  echo 'BOOT_ORDER=0xf416' >> /tmp/ee.txt
fi
grep -q '^PCIE_PROBE=' /tmp/ee.txt || echo 'PCIE_PROBE=1' >> /tmp/ee.txt
S rpi-eeprom-config --apply /tmp/ee.txt
echo "  EEPROM quedará:"
grep -E 'BOOT_ORDER|PCIE_PROBE' /tmp/ee.txt

echo
echo "LISTO. Reinicia con:  sudo reboot"
echo "Tras reiniciar, la raíz debe verse en /dev/nvme0n1p2 (comando: findmnt -n -o SOURCE /)"
echo "Si algo falla, apaga, saca el SSD del shield y la Pi vuelve a arrancar desde la microSD."
