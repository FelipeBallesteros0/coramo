# Guía: Entrenar wake word "coramo" en PC con Windows

Este proceso genera y entrena un modelo custom de openWakeWord para detectar la
palabra "coramo" en tiempo real (en CPU, sin GPU). El entrenamiento se hace en el
PC porque el RPi5 no tiene suficiente potencia de fuente para la carga de CPU sostenida.

**Tiempo estimado:** 2-4 horas en un PC moderno
**Resultado:** `coramo.tflite` (+ `coramo.onnx` de respaldo) para copiar al RPi5

---

## Requisitos previos

- Windows 10/11 con WSL2 instalado (Ubuntu 22.04 o 24.04)
- **Python 3.10 exactamente** — las dependencias de openWakeWord no son compatibles con 3.11/3.12
- ~5 GB de espacio libre
- Conexión a internet (para descargar datasets de HuggingFace)

> **¿Por qué WSL2 y no Python nativo de Windows?**
> `piper-tts` usa `espeak-ng` internamente, que requiere instalación extra en Windows.
> En WSL2 funciona igual que en Linux sin complicaciones.

---

## Paso 1: Activar WSL2 (si no lo tienes)

Abre PowerShell **como administrador** y ejecuta:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia el PC. Al volver, Ubuntu se abrirá y pedirá crear un usuario.

Si ya tienes WSL2, abre "Ubuntu" desde el menú inicio.

---

## Paso 2: Instalar Python 3.10 y dependencias del sistema en WSL2

Ubuntu 24.04 viene con Python 3.12 por defecto, pero openWakeWord necesita Python 3.10.
Instala ambas cosas con:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev espeak-ng git
```

Verifica que quedó instalado:

```bash
python3.10 --version   # debe mostrar Python 3.10.x
```

---

## Paso 3: Clonar el repositorio Coramo

```bash
cd ~
git clone https://github.com/TU_USUARIO/coramo.git
cd coramo
```

> Reemplaza `TU_USUARIO` con tu usuario de GitHub.

---

## Paso 4: Crear entorno virtual con Python 3.10 e instalar dependencias

```bash
python3.10 -m venv ~/train-env
source ~/train-env/bin/activate

pip install --upgrade pip
pip install -r training/requirements_train.txt
```

> La instalación puede tardar 10-15 minutos (descarga torch, tensorflow, etc.)

---

## Paso 5: Descargar el modelo de voz Piper en español

```bash
mkdir -p ~/piper-voices
cd ~/piper-voices

# Modelo es_ES-davefx (el mismo que usa el RPi)
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"

ls -la  # debe mostrar los dos archivos
```

---

## Paso 6: Ejecutar el entrenamiento

```bash
cd ~/coramo
source ~/train-env/bin/activate

python training/train_coramo.py \
  --piper-model ~/piper-voices/es_ES-davefx-medium.onnx \
  --output-dir ~/coramo_training \
  --n-samples 5000 \
  --n-val 1000 \
  --steps 10000
```

### ¿Qué hace el script?

El proceso tiene 7 fases que se ejecutan automáticamente:

| Fase | Descripción | Tiempo aprox. |
|------|-------------|---------------|
| 1 | Descarga `validation_set_features.npy` de HuggingFace (176 MB) | 2-5 min |
| 2 | Descarga 270 archivos de respuestas al impulso MIT | 5-10 min |
| 3 | Genera 50 clips de ruido de fondo sintético | < 1 min |
| 4 | Genera 12,000 muestras de voz con Piper (positivas + negativas) | 45-90 min |
| 5 | Augmenta clips y calcula features (openWakeWord) | 30-60 min |
| 6 | Entrena el modelo DNN (10,000 pasos) | 30-60 min |
| 7 | Exporta a `.onnx` y `.tflite` | < 5 min |

> El script **retoma donde se quedó** si lo interrumpes y vuelves a ejecutar —
> no repite fases ya completadas.

### Salida esperada al final:

```
[HH:MM:SS] INFO   ENTRENAMIENTO COMPLETADO
[HH:MM:SS] INFO   coramo.onnx  (XXX KB) -> ~/coramo_training/models/coramo.onnx
[HH:MM:SS] INFO   coramo.tflite (XXX KB) -> ~/coramo_training/models/coramo.tflite
[HH:MM:SS] INFO   Copia el modelo al RPi:
[HH:MM:SS] INFO     scp ~/coramo_training/models/coramo.tflite felipe@coramo.local:/home/felipe/coramo/models/
```

---

## Paso 7: Copiar el modelo al RPi5

Desde WSL2 (asegúrate que el RPi esté encendido y en la misma red):

```bash
scp ~/coramo_training/models/coramo.tflite felipe@coramo.local:/home/felipe/coramo/models/
```

Si `coramo.local` no resuelve, usa la IP directamente:

```bash
# Primero busca la IP del RPi en el router, luego:
scp ~/coramo_training/models/coramo.tflite felipe@192.168.X.X:/home/felipe/coramo/models/
```

Verificar que llegó:

```bash
ssh felipe@coramo.local "ls -la /home/felipe/coramo/models/"
```

---

## Paso 8: Activar openWakeWord en el asistente

Una vez copiado el modelo, en el RPi actualiza el código del asistente.
El script `coramo-assistant.py` ya tiene los cambios para usar openWakeWord
en el branch/commit correspondiente — solo actualiza y reinicia.

---

## Opciones avanzadas del script

```bash
# Si ya generaste los samples y solo quieres re-entrenar:
python training/train_coramo.py \
  --piper-model ~/piper-voices/es_ES-davefx-medium.onnx \
  --output-dir ~/coramo_training \
  --skip-generate

# Si ya tienes features y solo quieres re-entrenar con más pasos:
python training/train_coramo.py \
  --piper-model ~/piper-voices/es_ES-davefx-medium.onnx \
  --output-dir ~/coramo_training \
  --skip-generate \
  --skip-augment \
  --steps 25000

# Usar GPU (si tienes NVIDIA con CUDA):
python training/train_coramo.py \
  --piper-model ~/piper-voices/es_ES-davefx-medium.onnx \
  --output-dir ~/coramo_training \
  --device cuda
```

---

## Problemas comunes

### Error: `espeak-ng not found`
```bash
sudo apt install -y espeak-ng
```

### Error: `No module named 'piper'`
```bash
source ~/train-env/bin/activate  # Activa el entorno virtual primero
```

### El modelo tiene muchos falsos positivos (se activa solo)
Entrena de nuevo con más pasos:
```bash
python training/train_coramo.py --piper-model ... --skip-generate --skip-augment --steps 25000
```

### El modelo no detecta "coramo" (muchos falsos negativos)
Baja el threshold en `coramo-assistant.py`:
```python
OWW_THRESHOLD = 0.3  # prueba con 0.3 en lugar de 0.5
```

---

## Estructura de archivos generados

```
~/coramo_training/
├── clips/
│   ├── positive_train/     # ~5000 WAV de "coramo", "hola coramo", etc.
│   ├── positive_test/      # ~1000 WAV de validacion
│   ├── negative_train/     # ~5000 WAV de palabras similares
│   └── negative_test/      # ~1000 WAV de validacion
├── background/             # 50 clips de ruido sintetico
├── rirs/                   # 270 Room Impulse Responses (MIT)
├── features/               # Features pre-computadas (.npy)
├── validation_set_features.npy  # Dataset de validacion HuggingFace
└── models/
    ├── coramo.onnx         ← modelo listo para inferencia
    └── coramo.tflite       ← version TFLite (preferida para RPi)
```
