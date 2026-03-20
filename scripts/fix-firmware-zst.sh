#!/bin/bash
# Descomprimir firmwares .zst para kernel Coreforge 6.6.x
# El kernel 6.6.70-v8-16k+ no descomprime .zst automáticamente.
# Ejecutar después de cada actualización de linux-firmware.

set -e

DIRS=(
    /lib/firmware
    /lib/firmware/brcm
    /lib/firmware/amdgpu
    /lib/firmware/mediatek
    /lib/firmware/rtl_nic
)

total=0
for dir in "${DIRS[@]}"; do
    files=$(ls "$dir"/*.zst 2>/dev/null | wc -l)
    if [ "$files" -gt 0 ]; then
        echo "[$dir] Descomprimiendo $files archivos..."
        for f in "$dir"/*.zst; do
            zstd -d "$f" -o "${f%.zst}" --force -q
            ((total++))
        done
    fi
done

echo "Total descomprimidos: $total"
