#!/home/felipe/coramo-env/bin/python3
"""
Graba muestras de voz real para reentrenamiento de coramo.
Uso: python3 record_samples.py --output-dir ~/real_samples --count 60
Di "coramo" cuando veas [REC], espera a que aparezca el siguiente.
Ctrl+C para detener.
"""
import os
import sys
import time
import wave
import argparse
import subprocess

SAMPLE_RATE  = 16000
DURATION_SEC = 1.5   # duración de cada muestra
PRE_DELAY    = 0.8   # pausa antes de cada REC (tiempo para prepararse)


def record_one(path, device="default"):
    n_frames = int(SAMPLE_RATE * DURATION_SEC)
    proc = subprocess.Popen([
        "arecord", "-q", "-D", device,
        "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1",
        "--duration", str(DURATION_SEC), "-t", "raw",
    ], stdout=subprocess.PIPE)
    raw = proc.stdout.read(n_frames * 2)
    proc.wait()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Graba muestras de voz real para reentrenar coramo")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/real_samples"),
                        help="Directorio de salida (default: ~/real_samples)")
    parser.add_argument("--count", type=int, default=60,
                        help="Número de muestras a grabar (default: 60)")
    parser.add_argument("--device", default="default",
                        help="Dispositivo ALSA (default: default)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith(".wav")])

    print(f"Directorio: {args.output_dir}")
    print(f"Ya existen: {existing} muestras")
    print(f"A grabar: {args.count} muestras nuevas")
    print(f"\nCada muestra dura {DURATION_SEC}s. Di 'coramo' claro cuando aparezca [REC].")
    print("Varía un poco la entonación — rápido, lento, cerca, lejos del mic.")
    print("Ctrl+C para detener en cualquier momento.\n")
    time.sleep(1)

    for i in range(existing, existing + args.count):
        path = os.path.join(args.output_dir, f"real_{i:04d}.wav")
        time.sleep(PRE_DELAY)
        print(f"  [{i+1}/{existing + args.count}] [REC] di 'coramo'...", end="", flush=True)
        record_one(path, args.device)
        print(f" ok")

    total = existing + args.count
    print(f"\n✓ {args.count} muestras grabadas ({total} total) en {args.output_dir}")
    print(f"\nCopia al PC con (desde WSL2):")
    print(f"  scp -r felipe@coramo.local:{args.output_dir} ~/real_samples")


if __name__ == "__main__":
    main()
