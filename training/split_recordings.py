#!/usr/bin/env python3
"""
Convierte grabaciones .m4a a clips WAV individuales para reentrenamiento.
Cada archivo puede tener varias repeticiones de "coramo" — las separa por silencio.

Uso:
    python training/split_recordings.py \
        --input-dir "/mnt/c/Users/User/Documents/Grabaciones de sonido" \
        --output-dir ~/real_samples
"""
import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np

SAMPLE_RATE    = 16000
MIN_CLIP_SEC   = 0.4    # clip minimo para ser valido
MAX_CLIP_SEC   = 2.5    # clip maximo (corta si es mas largo)
SILENCE_DB     = -35    # umbral de silencio en dB
SILENCE_SEC    = 0.25   # duracion minima de silencio para separar clips
PAD_SEC        = 0.1    # padding antes y despues de cada clip


def convert_to_wav(src, dst):
    """Convierte cualquier formato de audio a WAV 16kHz mono."""
    r = subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", dst
    ], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo:\n{r.stderr.decode()}")


def detect_speech_segments(wav_path):
    """Detecta segmentos de habla usando ffmpeg silencedetect."""
    cmd = [
        "ffmpeg", "-i", wav_path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_SEC}",
        "-f", "null", "-"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    output = r.stderr

    # Parsear silence_start y silence_end
    silences = []
    starts, ends = [], []
    for line in output.splitlines():
        if "silence_start" in line:
            t = float(line.split("silence_start: ")[1].split()[0])
            starts.append(t)
        if "silence_end" in line:
            t = float(line.split("silence_end: ")[1].split("|")[0].strip())
            ends.append(t)

    # Obtener duracion total
    duration = None
    for line in output.splitlines():
        if "Duration" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
            duration = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            break

    if duration is None:
        return []

    # Construir segmentos de habla (entre silencios)
    boundaries = [0.0]
    for s, e in zip(starts, ends):
        boundaries.append(s)
        boundaries.append(e)
    boundaries.append(duration)

    segments = []
    for i in range(0, len(boundaries)-1, 2):
        t_start = boundaries[i]
        t_end   = boundaries[i+1]
        if (t_end - t_start) >= MIN_CLIP_SEC:
            segments.append((
                max(0.0, t_start - PAD_SEC),
                min(duration, t_end + PAD_SEC)
            ))

    return segments


def extract_clip(wav_path, t_start, t_end, out_path):
    """Extrae un segmento de audio."""
    duration = min(t_end - t_start, MAX_CLIP_SEC)
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ss", str(t_start), "-t", str(duration),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        out_path
    ], capture_output=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Divide grabaciones en clips individuales")
    parser.add_argument("--input-dir",  required=True, help="Directorio con archivos .m4a")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/real_samples"),
                        help="Directorio de salida para clips WAV (default: ~/real_samples)")
    args = parser.parse_args()

    input_dir  = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Buscar archivos de audio
    exts = {".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac"}
    files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not files:
        print(f"No se encontraron archivos de audio en {input_dir}")
        sys.exit(1)

    print(f"Encontrados {len(files)} archivo(s) en {input_dir}")
    print(f"Salida: {output_dir}\n")

    # Contar clips existentes
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    clip_idx = existing
    total_clips = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, src in enumerate(files):
            fname = os.path.basename(src)
            print(f"[{i+1}/{len(files)}] {fname}")

            # Convertir a WAV temporal
            tmp_wav = os.path.join(tmpdir, f"tmp_{i}.wav")
            try:
                convert_to_wav(src, tmp_wav)
            except RuntimeError as e:
                print(f"  ERROR convirtiendo: {e}")
                continue

            # Detectar segmentos de habla
            segments = detect_speech_segments(tmp_wav)
            if not segments:
                # Si no detecta silencios, usar el archivo completo como un clip
                out = os.path.join(output_dir, f"real_{clip_idx:04d}.wav")
                extract_clip(tmp_wav, 0, 99999, out)
                print(f"  -> 1 clip (sin silencios detectados): {os.path.basename(out)}")
                clip_idx += 1
                total_clips += 1
            else:
                print(f"  -> {len(segments)} segmento(s) detectados")
                for j, (t_start, t_end) in enumerate(segments):
                    out = os.path.join(output_dir, f"real_{clip_idx:04d}.wav")
                    try:
                        extract_clip(tmp_wav, t_start, t_end, out)
                        dur = t_end - t_start
                        print(f"     clip {j+1}: {t_start:.2f}s - {t_end:.2f}s ({dur:.2f}s) -> {os.path.basename(out)}")
                        clip_idx += 1
                        total_clips += 1
                    except subprocess.CalledProcessError as e:
                        print(f"     ERROR extrayendo clip {j+1}: {e}")

    print(f"\n✓ {total_clips} clips nuevos guardados en {output_dir}")
    print(f"  Total en directorio: {clip_idx}")
    print(f"\nProximo paso — reentrenar:")
    print(f"  source ~/train-env/bin/activate")
    print(f"  rm ~/coramo_training/features/positive_features_*.npy")
    print(f"  python training/train_coramo.py \\")
    print(f"    --piper-model ~/piper-voices/es_ES-davefx-medium.onnx \\")
    print(f"    --output-dir ~/coramo_training \\")
    print(f"    --real-recordings-dir {output_dir} \\")
    print(f"    --skip-generate --steps 10000")


if __name__ == "__main__":
    main()
