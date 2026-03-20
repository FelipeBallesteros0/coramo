#!/bin/bash
# Verificar estado de red y failover

echo "=== INTERFACES ==="
nmcli device status

echo ""
echo "=== RUTAS ==="
ip route show

echo ""
echo "=== INTERNET (eth0) ==="
if ip link show eth0 | grep -q "UP"; then
    ping -c 2 -I eth0 8.8.8.8 2>/dev/null | tail -2 || echo "eth0 sin ruta"
else
    echo "eth0 sin cable"
fi

echo ""
echo "=== INTERNET (WiFi USB) ==="
ping -c 2 -I wlx90de80052ea8 8.8.8.8 2>/dev/null | tail -2 || echo "WiFi USB no disponible"

echo ""
echo "=== INTERNET (general) ==="
ping -c 2 8.8.8.8 | tail -2
