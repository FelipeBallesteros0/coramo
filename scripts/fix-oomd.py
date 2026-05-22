#!/usr/bin/env python3
"""Fix systemd-oomd agresivo que causa reboots durante la carga del LLM."""
import subprocess
import sys

def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")

print("=== Fix OOM: deshabilitar systemd-oomd + aumentar swappiness ===\n")

print("[1] Maskeando systemd-oomd (socket + service) para que no pueda arrancar...")
run(["sudo", "systemctl", "stop", "systemd-oomd.socket", "systemd-oomd.service"])
run(["sudo", "systemctl", "mask", "systemd-oomd.socket", "systemd-oomd.service"])

print("[3] Aumentando swappiness a 80 (ahora)...")
run(["sudo", "sysctl", "vm.swappiness=80"])

print("[4] Haciendo swappiness permanente en /etc/sysctl.conf...")
with open("/etc/sysctl.conf", "r") as f:
    content = f.read()
if "vm.swappiness" not in content:
    run(["sudo", "bash", "-c", "echo 'vm.swappiness=80' >> /etc/sysctl.conf"])
else:
    print("  ya configurado, saltando.")

print("\n[OK] Listo. systemd-oomd deshabilitado, swappiness=80")
print("     El kernel OOM killer sigue activo como respaldo.")
print("     Ahora puedes correr: coramo")
