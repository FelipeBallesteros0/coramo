#!/usr/bin/env python3
"""
train_coramo.py — Entrena el modelo openWakeWord para la wake word "coramo".

Pipeline completa:
  1. Genera samples positivos ("coramo", "hola coramo", ...) con piper TTS
  2. Genera samples adversariales negativos (palabras similares en sonido)
  3. Augmenta los clips con ruido de fondo y respuestas al impulso
  4. Entrena el modelo DNN con openWakeWord
  5. Exporta coramo.onnx + coramo.tflite

Uso:
    python train_coramo.py --piper-model es_ES-davefx-medium.onnx [opciones]

Ver --help para todas las opciones.
"""

import os
import sys
import uuid
import random
import time
import wave
import io
import logging
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile

# ---------------------------------------------------------------------------
# Configurar logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_coramo")


# ---------------------------------------------------------------------------
# Frases objetivo y adversariales
# ---------------------------------------------------------------------------
POSITIVE_PHRASES = [
    "coramo",
    "hola coramo",
    "hey coramo",
    "oye coramo",
    "Coramo",
    "hola Coramo",
    "oye Coramo",
    "eh coramo",
    "mira coramo",
    "oiga coramo",
]

NEGATIVE_PHRASES = [
    # Foneticamente similares
    "cramo", "coralo", "corano", "corra", "cora", "coral", "colamo",
    "koramo", "curamo", "ceramo", "caramo",
    # Palabras comunes en espanol
    "como", "corona", "animo", "dinamo", "panorama", "diagrama",
    "hola", "hey", "oye", "para", "buenas",
    "hola cramo", "hey cramo", "oye corona", "hola corona",
    "colaron", "clamor", "llamar",
    "ramo", "drama", "trauma", "trama",
    "creo", "corro", "corres", "correr",
    "caro", "cura", "curas",
    "como estas", "que tal",
    "para el motor", "detente",
    "mueve el servo", "pon el motor",
    "uno dos tres",
    "por favor", "gracias", "de nada",
]

# Variaciones de velocidad y ruido de piper
LENGTH_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
NOISE_SCALES  = [0.3, 0.5, 0.667, 0.8, 1.0]
NOISE_WS      = [0.5, 0.667, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Paso 1: Generar samples con Piper
# ---------------------------------------------------------------------------
def generate_samples_piper(voice, phrases, output_dir, n_total, label):
    """Genera n_total WAV files usando la API Python de piper."""
    from piper.config import SynthesisConfig

    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    needed   = n_total - existing
    if needed <= 0:
        log.info(f"[{label}] Ya existen {existing} samples, saltando.")
        return

    log.info(f"[{label}] Generando {needed} samples en {output_dir} ...")
    rng       = random.Random(42 + hash(label))
    generated = 0
    start     = time.time()

    while generated < needed:
        text         = rng.choice(phrases)
        length_scale = rng.choice(LENGTH_SCALES)
        noise_scale  = rng.choice(NOISE_SCALES)
        noise_w      = rng.choice(NOISE_WS)

        cfg = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            for chunk in voice.synthesize(text, syn_config=cfg):
                wf.writeframes(chunk.audio_int16_bytes)

        wav_bytes = buf.getvalue()
        if len(wav_bytes) > 200:
            fname = os.path.join(output_dir, uuid.uuid4().hex + ".wav")
            with open(fname, "wb") as f:
                f.write(wav_bytes)
            generated += 1
            if generated % 500 == 0:
                elapsed = time.time() - start
                rate    = generated / elapsed
                eta_min = (needed - generated) / rate / 60
                log.info(f"  {generated}/{needed} | {rate:.1f}/s | ETA ~{eta_min:.0f} min")

    log.info(f"[{label}] Listo: {len(os.listdir(output_dir))} samples")


# ---------------------------------------------------------------------------
# Paso 2: Generar ruido de fondo sintetico (si no hay clips de fondo)
# ---------------------------------------------------------------------------
def generate_background_noise(out_dir, n_clips=50, sr=16000, duration=10):
    """Genera clips de ruido sintetico para augmentation."""
    import soundfile as sf

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    if existing >= n_clips:
        log.info(f"[background] Ya existen {existing} clips, saltando.")
        return

    log.info(f"[background] Generando {n_clips} clips de ruido en {out_dir} ...")
    rng = np.random.default_rng(42)
    generated = 0
    for i in range(n_clips):
        t = np.linspace(0, duration, sr * duration)
        kind = i % 5
        if kind == 0:
            audio = rng.normal(0, 0.001, sr * duration)
        elif kind == 1:
            audio = rng.normal(0, 0.05, sr * duration)
        elif kind == 2:
            uneven = rng.normal(0, 1, sr * duration)
            f = np.fft.rfftfreq(len(uneven))
            f[0] = 1
            pink = 1 / np.sqrt(f)
            audio = np.fft.irfft(np.fft.rfft(uneven) * pink)[: sr * duration]
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.1
        elif kind == 3:
            freq  = rng.choice([200, 400, 800, 1000, 2000])
            audio = 0.05 * np.sin(2 * np.pi * freq * t)
            audio += 0.02 * np.sin(2 * np.pi * freq * 2 * t)
            audio += rng.normal(0, 0.01, len(t))
        else:
            white = rng.normal(0, 1, sr * duration)
            audio = np.cumsum(white)
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.05

        fname = os.path.join(out_dir, f"background_{i:03d}.wav")
        sf.write(fname, audio.astype(np.float32), sr)
        generated += 1

    log.info(f"[background] Generados {generated} clips.")


# ---------------------------------------------------------------------------
# Paso 3: Augmentar clips y calcular features
# ---------------------------------------------------------------------------
def augment_and_compute_features(
    clips_dir, output_npy, total_length, background_paths, rir_paths,
    augmentation_rounds=2, batch_size=16, label="", device="cpu"
):
    """Augmenta clips de audio y computa features OWW, guardando en .npy"""
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    if os.path.exists(output_npy):
        log.info(f"[{label}] Features ya existen ({output_npy}), saltando.")
        return

    clips = [str(p) for p in Path(clips_dir).glob("*.wav")] * augmentation_rounds
    if not clips:
        raise RuntimeError(f"No hay WAV en {clips_dir}")

    log.info(f"[{label}] Augmentando {len(clips)} clips (rounds={augmentation_rounds})...")
    generator = augment_clips(
        clips,
        total_length=total_length,
        batch_size=batch_size,
        background_clip_paths=background_paths,
        RIR_paths=rir_paths,
    )

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    log.info(f"[{label}] Computando features OWW (device={device}, ncpu={n_cpus})...")
    compute_features_from_generator(
        generator,
        n_total=len(os.listdir(clips_dir)),
        clip_duration=total_length,
        output_file=output_npy,
        device=device,
        ncpu=n_cpus if device == "cpu" else 1,
    )
    log.info(f"[{label}] Features guardadas: {output_npy}")


# ---------------------------------------------------------------------------
# Paso 4: Entrenar modelo
# ---------------------------------------------------------------------------
def train_model(feature_dir, val_fp_path, steps, max_negative_weight, target_fp_per_hour, model_name, output_dir):
    """Entrena el DNN de openWakeWord y exporta ONNX."""
    from openwakeword.train import Model as OWWModel
    from openwakeword.data import mmap_batch_generator

    pos_train_npy = os.path.join(feature_dir, "positive_features_train.npy")
    neg_train_npy = os.path.join(feature_dir, "negative_features_train.npy")
    pos_test_npy  = os.path.join(feature_dir, "positive_features_test.npy")
    neg_test_npy  = os.path.join(feature_dir, "negative_features_test.npy")

    for p in [pos_train_npy, neg_train_npy, pos_test_npy, neg_test_npy]:
        if not os.path.exists(p):
            raise RuntimeError(f"Falta archivo de features: {p}")

    input_shape = np.load(pos_test_npy).shape[1:]
    log.info(f"[train] input_shape={input_shape}")

    oww = OWWModel(
        n_classes=1,
        input_shape=input_shape,
        model_type="dnn",
        layer_dim=32,
        seconds_per_example=1280 * input_shape[0] / 16000,
    )

    def f(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
        return x

    feature_data_files = {
        "positive":            pos_train_npy,
        "adversarial_negative": neg_train_npy,
    }
    data_transforms  = {k: f for k in feature_data_files}
    label_transforms = {
        "positive":             lambda x: [1 for _ in x],
        "adversarial_negative": lambda x: [0 for _ in x],
    }

    batch_generator = mmap_batch_generator(
        feature_data_files,
        n_per_class={"positive": 256, "adversarial_negative": 256},
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, gen):
            self.gen = gen
        def __iter__(self):
            return self.gen

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    X_train = torch.utils.data.DataLoader(
        IterDataset(batch_generator), batch_size=None, num_workers=n_cpus, prefetch_factor=16
    )

    X_val_fp_raw    = np.load(val_fp_path)
    X_val_fp_arr    = np.array([X_val_fp_raw[i:i+input_shape[0]] for i in range(0, X_val_fp_raw.shape[0]-input_shape[0], 1)])
    X_val_fp_labels = np.zeros(X_val_fp_arr.shape[0], dtype=np.float32)
    X_val_fp        = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp_arr), torch.from_numpy(X_val_fp_labels)),
        batch_size=len(X_val_fp_labels),
    )

    X_val_pos = np.load(pos_test_npy)
    X_val_neg = np.load(neg_test_npy)
    labels    = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)
    X_val     = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
            torch.from_numpy(labels),
        ),
        batch_size=len(labels),
    )

    log.info(f"[train] Iniciando entrenamiento ({steps} steps)...")
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=steps,
        max_negative_weight=max_negative_weight,
        target_fp_per_hour=target_fp_per_hour,
    )

    onnx_path = os.path.join(output_dir, model_name + ".onnx")
    log.info(f"[train] Exportando a {onnx_path}")
    oww.export_model(model=best_model, model_name=model_name, output_dir=output_dir)
    return onnx_path


# ---------------------------------------------------------------------------
# Paso 5: Convertir ONNX -> TFLite
# ---------------------------------------------------------------------------
def convert_to_tflite(onnx_path, tflite_path):
    """Convierte ONNX a TFLite para inferencia en RPi."""
    try:
        from openwakeword.utils import convert_onnx_to_tflite
        log.info(f"[tflite] Convirtiendo {onnx_path} -> {tflite_path}")
        convert_onnx_to_tflite(onnx_path, tflite_path)
        log.info(f"[tflite] Listo: {tflite_path}")
    except Exception as e:
        log.warning(f"[tflite] Conversion fallida (no critico): {e}")
        log.warning("[tflite] Usa el .onnx directamente — funciona igual en RPi.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrena wake word 'coramo' con openWakeWord")
    parser.add_argument("--piper-model",    required=True,  help="Ruta al modelo piper .onnx (es_ES-davefx-medium.onnx)")
    parser.add_argument("--output-dir",     default="./coramo_training", help="Directorio de salida (default: ./coramo_training)")
    parser.add_argument("--n-samples",      type=int, default=5000,  help="Samples de entrenamiento por clase (default: 5000)")
    parser.add_argument("--n-val",          type=int, default=1000,  help="Samples de validacion por clase (default: 1000)")
    parser.add_argument("--steps",          type=int, default=10000, help="Pasos de entrenamiento (default: 10000)")
    parser.add_argument("--augment-rounds", type=int, default=2,     help="Rondas de augmentation (default: 2)")
    parser.add_argument("--skip-generate",  action="store_true",     help="Saltar generacion de samples (si ya existen)")
    parser.add_argument("--skip-augment",   action="store_true",     help="Saltar augmentation (si features ya existen)")
    parser.add_argument("--device",         default="cpu",           help="Dispositivo para features: cpu o cuda (default: cpu)")
    parser.add_argument("--real-recordings-dir", default=None,
                        help="Directorio con grabaciones reales del usuario (.wav, 16kHz mono) para mezclar con sinteticos")
    args = parser.parse_args()

    # Rutas
    out        = os.path.abspath(args.output_dir)
    clips_dir  = os.path.join(out, "clips")
    pos_train  = os.path.join(clips_dir, "positive_train")
    pos_test   = os.path.join(clips_dir, "positive_test")
    neg_train  = os.path.join(clips_dir, "negative_train")
    neg_test   = os.path.join(clips_dir, "negative_test")
    bg_dir     = os.path.join(out, "background")
    rir_dir    = os.path.join(out, "rirs")
    feat_dir   = os.path.join(out, "features")
    val_fp_npy = os.path.join(out, "validation_set_features.npy")
    models_dir = os.path.join(out, "models")

    for d in [pos_train, pos_test, neg_train, neg_test, bg_dir, rir_dir, feat_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    log.info("=" * 60)
    log.info("  Entrenamiento openWakeWord — 'coramo'")
    log.info("=" * 60)
    log.info(f"  Output dir   : {out}")
    log.info(f"  Piper model  : {args.piper_model}")
    log.info(f"  Samples train: {args.n_samples} | val: {args.n_val}")
    log.info(f"  Steps        : {args.steps}")
    log.info(f"  Real samples : {args.real_recordings_dir or 'ninguno (solo sinteticos)'}")
    log.info("")

    # ---- PASO 0: Descargar modelos base de openWakeWord ----
    # melspectrogram.onnx y embedding_model.onnx son necesarios para calcular features
    from openwakeword.utils import download_models
    import openwakeword
    models_res_dir = os.path.join(
        os.path.dirname(openwakeword.__file__), "resources", "models"
    )
    melspec_path = os.path.join(models_res_dir, "melspectrogram.onnx")
    if not os.path.exists(melspec_path):
        log.info("[oww] Descargando modelos base (melspectrogram + embedding)...")
        download_models([])
        log.info("[oww] Modelos base listos.")
    else:
        log.info("[oww] Modelos base ya existen.")

    # ---- PASO 1: Descargar validation features de HuggingFace ----
    if not os.path.exists(val_fp_npy):
        log.info("[hf] Descargando validation_set_features.npy de HuggingFace (~176 MB)...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="davidscripka/openwakeword_features",
            filename="validation_set_features.npy",
            repo_type="dataset",
            local_dir=out,
        )
        log.info(f"[hf] Descargado: {val_fp_npy}")
    else:
        log.info(f"[hf] validation_set_features.npy ya existe.")

    # ---- PASO 2: Descargar MIT RIRs de HuggingFace ----
    rir_files = list(Path(rir_dir).glob("*.wav"))
    if len(rir_files) < 10:
        log.info("[hf] Descargando MIT Room Impulse Responses (~270 archivos)...")
        import soundfile as sf
        from datasets import load_dataset
        rir_dataset = load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train", streaming=True
        )
        count = 0
        for row in rir_dataset:
            audio = row["audio"]
            sf.write(os.path.join(rir_dir, f"rir_{count:05d}.wav"), audio["array"], audio["sampling_rate"])
            count += 1
        log.info(f"[hf] Descargados {count} RIRs.")
    else:
        log.info(f"[hf] RIRs ya existen ({len(rir_files)} archivos).")

    # ---- PASO 3: Generar ruido de fondo sintetico ----
    generate_background_noise(bg_dir, n_clips=50)

    # ---- PASO 4: Generar samples con Piper ----
    if not args.skip_generate:
        log.info(f"[piper] Cargando modelo {args.piper_model} ...")
        from piper.voice import PiperVoice
        voice = PiperVoice.load(args.piper_model)
        log.info("[piper] Modelo cargado.")

        generate_samples_piper(voice, POSITIVE_PHRASES, pos_train, args.n_samples, "positive/train")
        generate_samples_piper(voice, POSITIVE_PHRASES, pos_test,  args.n_val,     "positive/test")
        generate_samples_piper(voice, NEGATIVE_PHRASES, neg_train, args.n_samples, "negative/train")
        generate_samples_piper(voice, NEGATIVE_PHRASES, neg_test,  args.n_val,     "negative/test")
    else:
        log.info("[piper] Generacion omitida (--skip-generate).")

    # ---- PASO 4b: Copiar grabaciones reales del usuario ----
    if args.real_recordings_dir:
        real_dir = os.path.expanduser(args.real_recordings_dir)
        wavs = sorted([f for f in os.listdir(real_dir) if f.endswith(".wav")])
        if not wavs:
            log.warning(f"[real] No se encontraron .wav en {real_dir}, saltando.")
        else:
            n_train = int(len(wavs) * 0.8)
            train_wavs, test_wavs = wavs[:n_train], wavs[n_train:]
            copied = 0
            for i, fname in enumerate(train_wavs):
                dst = os.path.join(pos_train, f"real_train_{i:04d}.wav")
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(real_dir, fname), dst)
                    copied += 1
            for i, fname in enumerate(test_wavs):
                dst = os.path.join(pos_test, f"real_test_{i:04d}.wav")
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(real_dir, fname), dst)
                    copied += 1
            log.info(f"[real] {len(train_wavs)} train + {len(test_wavs)} test grabaciones reales copiadas ({copied} nuevas)")

    # ---- Calcular total_length desde samples positivos ----
    pos_test_clips = list(Path(pos_test).glob("*.wav"))
    if not pos_test_clips:
        log.error(f"No hay WAV en {pos_test}. Ejecuta sin --skip-generate primero.")
        sys.exit(1)

    sample_size = min(50, len(pos_test_clips))
    durations   = []
    for p in random.sample(pos_test_clips, sample_size):
        sr, data = scipy.io.wavfile.read(str(p))
        durations.append(len(data))

    total_length = int(round(np.median(durations) / 1000) * 1000) + 12000
    if total_length < 32000:
        total_length = 32000
    elif abs(total_length - 32000) <= 4000:
        total_length = 32000
    log.info(f"[config] total_length={total_length} samples ({total_length/16000:.2f}s)")

    # ---- PASO 5: Augmentar y calcular features ----
    rir_paths = [str(p) for p in Path(rir_dir).glob("*.wav")]
    bg_paths  = [str(p) for p in Path(bg_dir).glob("*.wav")]

    if not args.skip_augment:
        augment_and_compute_features(
            pos_train, os.path.join(feat_dir, "positive_features_train.npy"),
            total_length, bg_paths, rir_paths, args.augment_rounds, label="pos_train", device=args.device
        )
        augment_and_compute_features(
            neg_train, os.path.join(feat_dir, "negative_features_train.npy"),
            total_length, bg_paths, rir_paths, args.augment_rounds, label="neg_train", device=args.device
        )
        augment_and_compute_features(
            pos_test, os.path.join(feat_dir, "positive_features_test.npy"),
            total_length, bg_paths, rir_paths, args.augment_rounds, label="pos_test", device=args.device
        )
        augment_and_compute_features(
            neg_test, os.path.join(feat_dir, "negative_features_test.npy"),
            total_length, bg_paths, rir_paths, args.augment_rounds, label="neg_test", device=args.device
        )
    else:
        log.info("[augment] Augmentation omitida (--skip-augment).")

    # ---- PASO 6: Entrenar ----
    onnx_path = train_model(
        feature_dir=feat_dir,
        val_fp_path=val_fp_npy,
        steps=args.steps,
        max_negative_weight=500,
        target_fp_per_hour=0.5,
        model_name="coramo",
        output_dir=models_dir,
    )

    # ---- PASO 7: Convertir a TFLite ----
    tflite_path = onnx_path.replace(".onnx", ".tflite")
    convert_to_tflite(onnx_path, tflite_path)

    # ---- Resumen ----
    log.info("")
    log.info("=" * 60)
    log.info("  ENTRENAMIENTO COMPLETADO")
    log.info("=" * 60)
    for fname in ["coramo.onnx", "coramo.tflite"]:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            log.info(f"  {fname} ({size:.0f} KB) -> {fpath}")

    log.info("")
    log.info("  Copia el modelo al RPi:")
    log.info(f"    scp {models_dir}/coramo.tflite felipe@coramo.local:/home/felipe/coramo/models/")
    log.info("  (o usa coramo.onnx si tflite fallo)")


if __name__ == "__main__":
    main()
