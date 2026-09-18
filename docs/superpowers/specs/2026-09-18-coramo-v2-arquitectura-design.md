# CORAMO v2 — Diseño maestro de arquitectura

Fecha: 2026-09-18
Estado: borrador aprobado por secciones en conversación; pendiente de revisión final por Felipe.
Autor: Felipe Ballesteros León (con asistencia de Claude).

Este documento es la fuente de verdad del rediseño de CORAMO. Los subproyectos
A, B, C y D tendrán cada uno su propio spec y plan de implementación derivados de
este documento. Nada de lo que aquí se decide se reabre en los specs hijos sin
actualizar primero este archivo.

---

## 1. Propósito

CORAMO (**CO**laborativo **R**eprogramable **A**utónomo **MO**dular) es un robot
humanoide modular de tamaño real controlado por voz, con IA local o en la nube
según lo que resulte más rápido.

**Propósito de v2:** rehacer el proyecto desde cero con el conocimiento adquirido
en 2026, sin arrastrar los errores de v1, y ordenado por lo que evalúa un trabajo
de título en la UTEM (Escuela de Electrónica, profesor guía Patricio Galarce
Acevedo). El objetivo académico fijado por el profesor es:

> un robot que pueda interactuar con la gente y recibir órdenes.

Todo lo que se planifica aquí debe contribuir a demostrar y **medir** esa frase.
Lo comercial (CORAMO como plataforma de Nabla Robotics) viene después y se
beneficia de que v2 sea modular y documentado, pero no dicta prioridades.

No hay fecha de defensa todavía. El plan se organiza por hitos, no por calendario.

---

## 2. Punto de partida: v1 y sus lecciones

### 2.1 Qué existe (septiembre 2026)

| Elemento | Estado |
|---|---|
| Cuerpo físico | Torso, cabeza con soporte para 2 cámaras CSI, **un brazo** con motores DC + encoders AS5600, **mano** de 5 dedos con servos SG90. Perfiles de aluminio 20×20 y PLA. |
| Electrónica del brazo | Raspberry Pi Pico 2 W → PCA9685 → BTS7960 → motores DC. Encoders AS5600 por multiplexor I2C (TCA9548A o equivalente). Firmware MicroPython parcial (protocolo de 24 valores, "sin hardware"). |
| Mano | Servos SG90 por PCA9685. En v1 los manejaba un Arduino Mega; **en v2 no hay Arduino Mega**: todo cuelga del Pico. |
| Cerebro v1 | Raspberry Pi 5 + 2× RX 580 por multiplexor PCIe x1. Pipeline VAD → Whisper small → wake word por texto → Qwen3-4B con tools → Arduino / Piper. Latencia 6 s (83 % en Whisper por el bus). |
| Cerebro v2 (hardware) | Xeon E5-2670 + ASUS P9X79, 64 GB DDR3, RTX 4070 SUPER (12 GB) + RX 580 (8 GB), Ubuntu instalado, SSH en LAN. **Apagado por ahora**; Felipe lo enciende cuando empiece la tarea cero. |
| Simulación | `mujoco_test/` con un brazo antropomórfico genérico de 7 GDL y una base móvil; no describe el brazo real. |
| Repositorio | `github.com/FelipeBallesteros0/coramo`, último commit 2026-05-22. |
| Documentación | README, `docs/01..06`, coramo.cl (abril 2026), paper IEEE borrador, informe LaTeX, slides. Contradictorias entre sí. |
| Otro robot | El torso InMoov del repo `inmoov-voice-assistant` (julio 2026) es **otro robot**, no CORAMO. Su código sí sirve de referencia por ser modular y con tests. |

### 2.2 Errores de v1 que v2 no repite

1. **Plataforma exótica antes que robot.** Meses en kernel Coreforge, preload de
   `memcpy`, timeouts de amdgpu. El bus PCIe x1 fijó la latencia. → v2 corre en
   x86 con PCIe x16 real y drivers de fábrica.
2. **Modelos elegidos por lo que aguantaba la GPU.** Whisper small en fp32 por
   falta de fp16 en Vulkan; Qwen3-4B con contexto 1024. → v2 elige modelos por lo
   que necesita el robot y los corre en CUDA.
3. **Monolito sin tests.** `coramo-assistant.py` (740 líneas) mezclaba audio,
   VAD, STT, LLM, TTS y serial; configuración por variables de entorno. → v2 tiene
   paquetes con núcleo puro y tests; configuración en YAML.
4. **Tres protocolos para mover motores** (JSON al Mega, texto al Mega InMoov,
   24 valores al Pico). → v2 tiene **un** protocolo versionado.
5. **Sin seguridad real**: solo la lista cerrada de tools. → v2 tiene watchdog,
   límites, detección de sobrecorriente y parada física, todo en el micro.
6. **Whisper como subprocess** que recargaba el modelo por llamada (por un bug
   de whisper-server en Vulkan). → v2 mantiene todos los modelos residentes.
7. **Wake word entrenada y abandonada** (openWakeWord) dejada como peso muerto.
   → v2 no la incluye.
8. **Documentación contradictoria** (Reconfigurable vs Reprogramable, RPi5 vs
   Xeon 32 GB vs 64 GB, 5 objetivos vs 6 fases). → v2 tiene una sola fuente de
   verdad en `docs/`.
9. **Simulación desconectada del robot real.** → v2 deriva el modelo del CAD.

### 2.3 Lecciones de otros proyectos de 2026 que v2 aprovecha

- **Traductor de Discord (4070 SUPER):** faster-whisper large-v3-turbo 0,12 s,
  Kokoro en GPU 0,19 s por frase (0,65 s en CPU), compuerta de −45 dBFS RMS
  para que Whisper no alucine con ruido de sala. onnxruntime-gpu debe ser la
  versión de CUDA 12 (1.29.0 del índice de NVIDIA), no la de PyPI (CUDA 13).
- **Robot 4WD (ROS 2 Jazzy en RPi5):** `slam-toolbox`, `foxglove-bridge`,
  servicio systemd. Trampas: descubrimiento DDS por multicast falla si `lo` no
  tiene el flag MULTICAST; la memoria compartida de FastDDS queda corrupta tras
  un kill; `serial.write()` de pyserial se cuelga para siempre sin
  `write_timeout` cuando el CDC-ACM se atasca. Rieles de alimentación separados
  entre lógica y motores, con tierra común.
- **Repo InMoov:** estructura `voice_assistant/` + `tests/` con 13 archivos de
  prueba; validación de articulaciones y límites antes de escribir por serial.
- **Volante FFB:** Pico con CAN y protocolo serial; experiencia con el SDK de
  Pico y con ODrive.

---

## 3. Alcance

### 3.1 Entra en v2

- **Cerebro (A):** pipeline voz→acción nuevo, modular, con tests, en la 4070.
- **Protocolo y firmware del cuerpo (B):** un protocolo cerebro↔Pico con
  seguridad; firmware C++ del Pico para mano, brazo y cabeza.
- **Electrónica y control del brazo (C):** inventario, sensado de corriente,
  lazo de posición, calibración, URDF del brazo real.
- **Visión e interacción (D):** cámaras CSI en la RPi5 como nodo cabeza,
  detección de personas, mirar y saludar, recibir órdenes.

### 3.2 Fuera de v2 (va al capítulo "trabajo futuro" de la tesis)

Segundo brazo, piernas y locomoción, aprendizaje por demostración,
actualizaciones OTA, integración IoT, soporte multilingüe, manipulación guiada
por visión con agarre de objetos, control de impedancia.

### 3.3 Decisiones ya tomadas (no se reabren)

| Decisión | Elección | Razón |
|---|---|---|
| Columna vertebral | **ROS 2 Jazzy** | Felipe lo domina (4WD); Foxglove, `/joint_states`, TF y MoveIt disponibles; una toolchain para los dos robots de Nabla; reconocible por una comisión. |
| IA | **Local o nube, por latencia medida** | Decisión de Felipe (2026-09-18): lo que importa es la velocidad. Backends intercambiables (sección 5.6); los locales quedan como respaldo; la seguridad nunca depende de la red. |
| GPU de inferencia | **RTX 4070 SUPER** para todo lo crítico | Una sola toolchain (CUDA). |
| RX 580 | **Pantalla y reserva** | Evita duplicar toolchains (Vulkan/ROCm). |
| Firmware del Pico | **C/C++ con el SDK oficial** | Lazo de control determinista de varios ejes; MicroPython no lo da. |
| PWM de motores | **Se mantiene PCA9685 → BTS7960** | Decisión de Felipe para no recablear. Restricción conocida: PWM tope 1,5 kHz (silbido audible, control de corriente más grueso). |
| Cámaras | **CSI en la RPi5 de v1 como nodo cabeza** | Decisión de Felipe para reutilizar hardware. Obliga a DDS entre dos máquinas (sección 8). |
| Repositorio | **Mismo repo**, `main` reinicia limpio | v1 se conserva en rama `v1-rpi5` y tag `v1.0`. |
| Configuración | **YAML por perfil** | Reemplaza las variables de entorno de v1. |

---

## 4. Hardware objetivo

### 4.1 Cerebro: Xeon E5-2670 + P9X79

- 8 núcleos / 16 hilos Sandy Bridge-EP, 64 GB DDR3 cuádruple canal, PCIe 3.0.
- **Sin AVX2** (solo AVX). PyTorch, CTranslate2 y llama.cpp funcionan con sus
  rutas de respaldo, pero **ninguna inferencia se planifica en CPU**. Silero VAD
  y el remuestreo de audio sí corren en CPU sin problema.
- RTX 4070 SUPER en slot x16: inferencia (CUDA). Sin monitor conectado.
- RX 580 en el otro slot x16: pantalla del escritorio y Foxglove local. Se fija
  como GPU primaria en BIOS para que la 4070 quede libre.
- Fuente: 4070 SUPER (220 W) + RX 580 (185 W) + Xeon (115 W) exigen **≥ 750 W**.
  Verificar en la tarea cero.
- Audio: micrófono USB y altavoz USB conectados al Xeon (no a la cabeza), para
  mantener la latencia de voz en una sola máquina. Se puede mover a la cabeza
  en una versión posterior si la distancia lo exige.
- Conexión al cuerpo: **USB** al Pico. El Pico no usa WiFi en v2.

### 4.2 Cabeza: Raspberry Pi 5 de v1

- Se retiran el multiplexor X1011, las RX 580 y el kernel Coreforge. Se
  reinstala **Ubuntu 24.04 estándar para RPi** (kernel raspi de fábrica, con
  soporte de cámaras CSI vía libcamera).
- Corre ROS 2 Jazzy con `camera_ros` publicando las dos cámaras CSI.
- Se une al Xeon por **cable Ethernet directo** con IP estática (red aislada
  del WiFi de la casa), no por WiFi.

### 4.3 Cuerpo: Pico 2 W y electrónica

- Pico 2 W (RP2350) conectado por USB al Xeon.
- PCA9685 #1: 12 canales para 6 motores DC (RPWM/LPWM por BTS7960).
- PCA9685 #2: servos SG90 de la mano (5) y servos de cabeza (2). El #1 solo tiene
  4 canales libres, insuficientes para 7 servos.
- BTS7960 × 6 con pines IS de corriente hacia un **ADS1115** (4 canales; se
  necesitan 2 ADS1115 o un multiplexado; se decide en el inventario de C).
- AS5600 × n (uno por articulación del brazo) tras un **TCA9548A**.
- Alimentación: riel lógico (Pico, PCA9685, ADS1115, AS5600) separado del riel
  de motores/servos, tierra común.
- **Botón de parada física** que corta el riel de motores, no el lógico. El
  Pico lee su estado por un GPIO.

El inventario exacto (número de motores, reducciones, tensión, corriente de
bloqueo, rangos angulares, mapa de canales) es la **primera tarea de C** y
alimenta el URDF.

---

## 5. Arquitectura de software

### 5.1 Repositorio y estructura

```
coramo/                       (rama main reiniciada; v1 en rama v1-rpi5, tag v1.0)
├── src/                      workspace ROS 2 (colcon)
│   ├── coramo_brain/         audio, VAD, STT, agente, TTS
│   │   ├── coramo_brain/core/    Python puro, sin rclpy, con tests
│   │   └── coramo_brain/nodes/   envoltorios rclpy sin lógica
│   ├── coramo_body/          protocolo, puente serial, seguridad
│   ├── coramo_vision/        detección de personas, seguimiento
│   ├── coramo_description/   URDF, meshes, límites articulares
│   ├── coramo_msgs/          mensajes y servicios propios
│   └── coramo_bringup/       launch, params/*.yaml, systemd
├── firmware/pico/            C++ con Pico SDK (CMake)
├── head/                     configuración de la RPi5 (instalación, launch)
├── tools/                    scripts de medición y calibración
├── docs/                     fuente de verdad (ver sección 11)
│   └── superpowers/specs/    este documento y los specs hijos
└── tests/                    pruebas de integración (con audio grabado, sin robot)
```

- Clon de trabajo: `~/coramo` en WSL. La carpeta `Desktop\coramo` sigue siendo
  solo CAD, presupuestos y tesis; no contiene código.
- **Regla anti-monolito:** toda la lógica vive en `core/` de cada paquete y se
  prueba con pytest sin ROS. Los nodos solo traducen mensajes ↔ llamadas.

### 5.2 Grafo ROS 2

Todos los nodos del cerebro corren en el Xeon. Solo `head_cameras` corre en la
RPi5.

| Nodo | Entrada | Salida | Máquina |
|---|---|---|---|
| `audio` | micrófono USB | `/audio/chunks` (16 kHz mono, 32 ms) | Xeon |
| `vad` | `/audio/chunks` | `/speech/segment` (audio de un turno completo) | Xeon |
| `stt` | `/speech/segment` | `/speech/text` | Xeon (4070) |
| `agent` | `/speech/text`, `/vision/people`, `/coramo/state` | `/body/command`, `/tts/say`, `/head/look_at` | Xeon (4070) |
| `tts` | `/tts/say` | altavoz USB, `/tts/status` | Xeon (4070) |
| `safety` | `/body/command`, `/head/look_at`, latidos de nodos | `/body/command_safe`, `/body/estop` | Xeon |
| `body_bridge` | `/body/command_safe` | `/joint_states`, `/body/telemetry`, servicio `/body/estop` | Xeon |
| `head_cameras` | 2 cámaras CSI | `/head/cam_left/image_raw/compressed`, `/head/cam_right/...` | RPi5 |
| `vision` | imágenes | `/vision/people` (posición, distancia estimada, mira al robot sí/no) | Xeon (4070) |
| `state` | eventos de los demás | `/coramo/state` | Xeon |
| `foxglove_bridge` | todo | websocket para Foxglove | Xeon |

Todo comando físico pasa por `safety` antes de llegar a `body_bridge`:
`body_bridge` solo escucha `/body/command_safe`. El agente y visión publican
en `/body/command` y `/head/look_at`, nunca en `/body/command_safe`.

### 5.3 Máquina de estados

```
IDLE ──(habla detectada)──► LISTENING ──(fin de turno)──► THINKING
  ▲                                                          │
  │                                       ┌──────────────────┤
  │                                       ▼                  ▼
  └──────────────── SPEAKING ◄──── ACTING ◄──── (tool elegida)
                       ▲                                     │
                       └────────── (responder) ◄─────────────┘
```

Estados: `IDLE`, `LISTENING`, `THINKING`, `ACTING`, `SPEAKING`, `STOPPED`
(parada activa; solo sale con reconocimiento explícito). Cada transición lleva
marca de tiempo: de ahí sale la tabla de latencias de la tesis sin
instrumentación extra.

### 5.4 Reparto de GPU

| Modelo | VRAM aprox. | GPU |
|---|---|---|
| faster-whisper large-v3-turbo (fp16, CTranslate2) | 1,5 GB | 4070 |
| Qwen3-8B Q5_K_M en llama-server, contexto 8k, KV q8 | 6,5 GB | 4070 |
| Kokoro (voz española) | 0,3 GB | 4070 |
| Detector de personas/caras (YOLO nano o MediaPipe) | 0,5 GB | 4070 |
| **Total** | **≈ 9 GB de 12** | |

Si en la tarea cero el total supera 11 GB, se baja el LLM a Q4_K_M antes de
tocar cualquier otra cosa. Los modelos locales se cargan aunque el perfil use
nube, porque son el respaldo (sección 5.6).

### 5.5 Configuración

Un YAML por perfil en `coramo_bringup/params/`:

- `xeon.yaml`: producción, robot conectado.
- `dev-sin-robot.yaml`: mismo cerebro, `body_bridge` en modo simulado (acepta
  comandos, publica `/joint_states` sintéticos). Permite desarrollar y correr
  tests sin el robot ni el servidor encendidos.
- `head.yaml`: la RPi5.

### 5.6 Backends de IA intercambiables

Decisión de Felipe (2026-09-18): **lo que importa es la velocidad, no que todo
sea local.** Cada etapa pesada (STT, LLM, TTS) es un backend intercambiable
detrás de una interfaz única en `core/`, y el YAML del perfil elige cuál se usa.
La elección se hace con benchmark en la tarea cero, no por preferencia.

| Etapa | Local (4070) | Nube (candidatos) | Regla de elección |
|---|---|---|---|
| STT | faster-whisper large-v3-turbo, 0,12 s medido en el traductor | Transcripción de OpenAI (ya usada en el repo InMoov y en el bot de WhatsApp) o un servicio de streaming | Menor latencia fin de habla → texto con WER ≤ 10 % |
| LLM | Qwen3-8B Q5 en llama-server | Claude Haiku 4.5 (`claude-haiku-4-5`, US$ 1 / 5 por millón de tokens de entrada / salida) o Claude Sonnet 5 (`claude-sonnet-5`, US$ 2 / 10); OpenAI o DeepSeek como alternativas ya usadas | Mejor acierto de tool con latencia ≤ 0,5 s. En la nube: `tool_choice` forzado a una tool, sin thinking, system prompt con cache |
| TTS | Kokoro, 0,19 s medido | TTS de OpenAI en streaming, ElevenLabs o Fish Audio (ya probado) | Menor tiempo al primer audio con voz española aceptable |
| Visión | Detector local, siempre | ninguno | 15 FPS continuos no se mandan a la nube |

Reglas fijas, independientes del backend:

- Los modelos locales quedan cargados aunque el perfil use nube: son el
  **respaldo automático** si la red falla o una petición supera el tiempo
  límite (2 s). Sin internet el robot sigue funcionando, más lento.
- La nube recibe solo el segmento de audio del turno y el texto; nunca audio
  continuo ni video.
- La cadena de seguridad (watchdog, botón, límites en el Pico) no depende de
  la red. El "detente" por voz hereda la latencia del STT en uso; el botón
  físico es la parada primaria.
- Costo por orden con el system prompt en cache: menos de un centavo de dólar
  con Haiku 4.5 o Sonnet 5. Se mide y se registra en `docs/mediciones/`.

---

## 6. Subproyecto A — Cerebro

### 6.1 Pipeline

1. `audio`: captura continua, 16 kHz mono, chunks de 32 ms, compuerta de nivel
   (−45 dBFS RMS, del traductor) para no alimentar ruido de sala.
2. `vad`: Silero VAD en CPU. Fin de turno tras 600 ms de silencio (ajustable),
   tope de 15 s. Publica el segmento completo con marcas de inicio y fin.
3. `stt`: backend según perfil (local faster-whisper large-v3-turbo con
   `language=es` y `beam_size=1`, o nube). El modelo local queda residente.
   Publica texto + confianza.
4. Wake word por texto con coincidencia difusa ("coramo", "hola coramo"...),
   **excepto en modo cara a cara**: si `/vision/people` reporta una persona a
   menos de ~2 m mirando al robot, no se exige la palabra.
5. `agent`: backend según perfil (llama-server con Qwen3-8B y `--jinja`, o
   Claude Haiku 4.5 / Sonnet 5 por API). Siempre una tool obligatoria por
   turno, temperatura 0, historial corto (últimos 6 turnos), system prompt con
   identidad y reglas, y cache del system prompt (KV local o prompt caching
   en la nube).
6. Tool elegida → `safety` → `body_bridge` (acción) y/o `tts`.
7. `tts`: backend según perfil (Kokoro local o TTS en nube), streaming por
   oración. Piper en CPU como último respaldo.

### 6.2 Tools expuestas al LLM

| Tool | Argumentos | Efecto |
|---|---|---|
| `mano` | `gesto` (abre, cierra, paz, ok, rock, pulgar) **o** `dedos` {nombre: ángulo} | comando a los servos de la mano |
| `brazo` | `pose` nombrada (reposo, saludo, extendido, ...) **o** `articulaciones` {nombre: grados} | comando al brazo, validado contra el URDF |
| `cabeza` | `mirar` (persona, frente, izquierda, derecha) o ángulos pan/tilt | comando a la cabeza |
| `responder` | `texto` | TTS |
| `detener` | — | parada: `/body/estop`, estado `STOPPED` |

Reglas: el LLM elige **una** tool por turno. Los nombres de articulaciones son
los del URDF. `safety` rechaza valores fuera de límite y responde por voz
"no puedo hacer eso" en vez de recortar en silencio. "Detente" / "para" se
detecta también **antes** del LLM, por texto, para no depender de la inferencia.

### 6.3 Latencia objetivo (fin de habla → primera acción)

"Fin de habla" es el instante en que el usuario deja de hablar. El silencio
que el VAD necesita para cerrar el turno **cuenta** dentro de la latencia,
porque el usuario lo percibe.

Objetivos por etapa, independientes del backend elegido:

| Etapa | Objetivo |
|---|---|
| Fin de turno (silencio VAD) | 0,6 s (es parte de la percepción del usuario) |
| STT | ≤ 0,3 s |
| LLM (≤ 40 tokens de tool call) | ≤ 0,5 s |
| Seguridad + serial | 0,02 s |
| **Acción física** | **≤ 1,5 s desde fin de habla (p95)** |
| Primer audio TTS | < 1,8 s |

Se mide con las marcas de `/coramo/state`, no se estima. Cada backend (local y
nube) se mide en la tarea cero con 30 peticiones (p50 y p95) sobre la red de
la casa; el ganador por etapa va a `xeon.yaml` y el resultado queda en
`docs/mediciones/`.

### 6.4 Pruebas

- Unitarias por módulo de `core/`: VAD sobre WAV grabados; wake word con
  variantes mal transcritas ("coramos", "coramó"); parser de tools con salidas
  reales del modelo; validación de límites.
- Set de evaluación: 50 a 100 órdenes grabadas en español por 3 o más voces,
  con y sin ruido. Métricas: WER del STT, acierto de tool y de argumentos.
- Integración sin robot con `dev-sin-robot.yaml`: audio grabado entra,
  `/body/command` esperado sale.

---

## 7. Subproyectos B y C — Cuerpo

### 7.1 Protocolo cerebro↔Pico v2

Transporte: USB CDC, 115200 o superior (CDC ignora el baudio), **JSON por
líneas** (una línea = un mensaje, terminada en `\n`). Campos comunes: `v`
(versión del protocolo, entero), `t` (tipo), `seq` (secuencia).

Del cerebro al Pico:

```json
{"v":2,"t":"cmd","seq":41,"joints":{"codo_flex":45.0,"indice":90.0},"mode":"pos"}
{"v":2,"t":"hb","seq":42}
{"v":2,"t":"estop","seq":43}
{"v":2,"t":"cfg","seq":44,"joints":{"codo_flex":{"min":0,"max":135,"kp":1.2,"ki":0.0,"kd":0.05,"imax":6.0}}}
{"v":2,"t":"home","seq":45}
```

Del Pico al cerebro:

```json
{"v":2,"t":"tel","seq":900,"st":"run","estop":false,
 "j":{"codo_flex":{"p":44.7,"i":1.3,"s":"ok"},"indice":{"p":90,"s":"ok"}}}
{"v":2,"t":"ack","seq":41}
{"v":2,"t":"err","seq":41,"code":"limit","joint":"codo_flex"}
```

- `mode`: `pos` (posición articular) en v2. Se reserva `vel` para después.
- `tel` a 50 Hz. `st` ∈ {`boot`, `run`, `hold`, `estop`, `fault`}.
- Estado por eje `s` ∈ {`ok`, `limit`, `overcurrent`, `sensor`}.
- Los nombres de articulación se cargan con `cfg` al conectar; el Pico rechaza
  comandos a nombres no configurados.
- JSON y no binario porque 6 ejes a 50 Hz son ~8 KB/s, se depura con `cat` y el
  formato es legible en la tesis. Si alguna vez hace falta, se agrega un modo
  binario COBS + CRC **con el mismo esquema**.

Lado cerebro: `pyserial` con `write_timeout` y `timeout` obligatorios (lección
del 4WD), hilo lector dedicado, reconexión automática si el dispositivo
desaparece.

### 7.2 Seguridad (vive en el Pico)

| Evento | Reacción |
|---|---|
| Sin `hb` durante 250 ms | estado `hold`: el PID mantiene posición, se rechazan comandos. Apagar motores haría caer el brazo por gravedad. Sale de `hold` con el siguiente `hb` + `cmd`. |
| Sobrecorriente sostenida en un eje (umbral e histéresis por `cfg`) | PWM de ese eje a cero, eje en `overcurrent`, estado global `fault`. Requiere `home` o reinicio explícito. |
| Lectura de encoder inválida (magnet fuera de rango, error I2C) | eje en `sensor`, PWM a cero en ese eje. |
| Botón de parada física | corta el riel de motores; el Pico lo detecta por GPIO, reporta `estop:true`, pasa a `estop`. Al soltar, no reanuda solo: espera `home`. |
| `estop` por software | PWM a cero en todos los ejes DC, servos liberan; estado `estop`. |
| Comando fuera de `min/max` | rechazado con `err` (no se recorta). |

### 7.3 Firmware del Pico 2 W

- C++ con el SDK oficial (CMake). Compilación en WSL, flasheo por UF2 o
  `picotool`.
- Núcleo 0: USB CDC, parser JSON (biblioteca ligera, sin asignación dinámica en
  el lazo), máquina de estados, watchdog de latido.
- Núcleo 1: lazo de control a **100 Hz**: selecciona canal del TCA9548A, lee cada
  AS5600, calcula PID por eje con anti-windup, escribe los 12 canales del
  PCA9685 en una sola transacción (autoincremento), lee corrientes del ADS1115.
  Presupuesto de bus a 400 kHz: ~2 ms encoders + ~2 ms PWM + ~1 ms corrientes,
  holgado para 10 ms de periodo.
- Servos de la mano y cabeza: PCA9685 a 50 Hz, posición directa (sin lazo).
- Watchdog de hardware del RP2350 activo: si el firmware se cuelga, reinicia en
  `boot` con PWM a cero.
- Sin WiFi ni Bluetooth en v2.

### 7.4 Electrónica y control del brazo (C)

1. **Inventario** (primera tarea): motores (modelo, tensión, reducción,
   corriente nominal y de bloqueo), sentido por eje, rango mecánico, canales de
   PCA9685 y del TCA9548A, ubicación de cada AS5600. Sale como
   `docs/hardware/brazo.md` + `coramo_description/config/joints.yaml`.
2. **Sensado de corriente**: pines IS del BTS7960 (≈ 8,5 kΩ de relación) →
   ADS1115. Se calibra con carga conocida.
3. **Calibración de cero**: procedimiento `home` que lleva cada eje a un tope
   mecánico conocido o a una marca, y guarda el offset del AS5600 en flash.
4. **Lazo de posición**: PID por eje con límites de PWM, rampa de velocidad para
   evitar tirones, feedforward de gravedad como mejora posterior si el error en
   reposo lo justifica.
5. **URDF**: `coramo_description` desde el CAD (existen STEP del codo y unión de
   hombro). Longitudes y límites reales. MuJoCo se usa solo para validar el
   control con este mismo modelo (conversión URDF → MJCF), reemplazando el brazo
   genérico de `mujoco_test/`.

Restricción conocida y aceptada: PWM a 1,5 kHz por el PCA9685. Se documenta y se
mide su efecto (ruido acústico, rizado de corriente). Si en C el control de
corriente resulta insuficiente, se reabre la decisión con datos.

### 7.5 Validación de B y C

- Desconectar USB con el brazo en una pose → `hold` en < 250 ms, sin caída.
- Bloquear un motor a mano → `fault` en ese eje sin afectar a los demás.
- Pulsar parada física → riel cortado, estado `estop`, y no reanuda solo.
- Rampa articular de 0→90° en 2 s en cada eje: error RMS y sobrepaso medidos y
  registrados en `docs/mediciones/`.
- Mano: los 6 gestos y control por dedo desde ROS, con telemetría en Foxglove.

---

## 8. Subproyecto D — Visión e interacción

### 8.1 Nodo cabeza (RPi5)

- Ubuntu 24.04 estándar para RPi5, ROS 2 Jazzy base, `camera_ros` (libcamera).
- Publica cada cámara a 640×480, 15 FPS, JPEG comprimido. Ancho de banda
  ≈ 1,2 MB/s por las dos: trivial por Ethernet.
- Solo captura y publica. Ninguna inferencia en la RPi5.
- Servicio systemd que arranca al encender, como el `mapper` del 4WD.

### 8.2 DDS entre dos máquinas (la parte que dolió en el 4WD)

- Cable Ethernet directo Xeon ↔ RPi5, IPs estáticas en una subred propia
  (p. ej. 192.168.50.1 / 192.168.50.2), sin pasar por el WiFi de la casa.
- **Fast DDS Discovery Server** en el Xeon; ambas máquinas con
  `ROS_DISCOVERY_SERVER` apuntando a él. Sin multicast, sin
  `ROS_LOCALHOST_ONLY`, sin adivinar interfaces.
- `chrony` en ambas con el Xeon como referencia, para que las marcas de tiempo
  de imagen y voz sean comparables.
- Verificación en la tarea cero: `ros2 topic hz` de las cámaras desde el Xeon
  estable durante 10 minutos.

### 8.3 Pipeline de visión (Xeon)

- Detector de personas y caras en la 4070 (YOLO nano o MediaPipe; se elige en
  la tarea cero por FPS y facilidad de instalación con CUDA 12).
- Seguimiento de la persona más cercana (estimación de distancia por altura de
  la caja o por disparidad estéreo en D completo).
- `/vision/people`: lista con posición en imagen, distancia estimada y
  "mira al robot" (cara frontal detectada).
- `/head/look_at`: la cabeza sigue a la persona con un lazo suave (servos de
  cabeza vía Pico).
- Objetivo: 15 FPS, latencia < 100 ms de imagen a `/vision/people`.

### 8.4 Comportamientos de interacción (lo que se demuestra en la tesis)

1. **Detectar y saludar**: persona nueva a < 2 m → la cabeza la mira, el robot
   saluda por voz y levanta la mano (gesto "saludo" del brazo cuando C esté).
2. **Conversar corto**: preguntas generales, identidad del robot, hora.
3. **Recibir orden**: gestos de mano, poses nombradas del brazo, mirar.
4. **"Detente" prioritario**: por voz (detección por texto antes del LLM) y por
   botón físico.
5. **"No entendí"**: respuesta verbal cuando la confianza del STT es baja o el
   LLM no encuentra tool aplicable; nunca inventa una acción.

---

## 9. Hitos y orden

| Hito | Contenido | Criterio de aceptación |
|---|---|---|
| **0 — Servidor listo** | Xeon encendido y verificado: Ubuntu 24.04, driver NVIDIA + CUDA 12, amdgpu como pantalla, ROS 2 Jazzy, Discovery Server, fuente. RPi5 reinstalada como cabeza. Modelos locales descargados y claves de API configuradas. Benchmark local vs nube por etapa. | Tabla de latencia por backend (STT, LLM, TTS, p50 y p95 de 30 peticiones) y FPS del detector, medidos. Tabla de decisión de backends. Cámaras visibles desde el Xeon. |
| **A — Cerebro** | Paquetes `coramo_brain`, `coramo_msgs`, `coramo_bringup`; `body_bridge` simulado. | "coramo, cierra la mano" → `/body/command_safe` en ≤ 1,5 s desde fin de habla, medido. Conversación básica. Tests verdes sin robot. |
| **B — Cuerpo (mano primero)** | Protocolo v2, firmware Pico C++, `body_bridge` real, `safety`. | Mano y cabeza comandadas desde ROS con telemetría en Foxglove. Watchdog y parada verificados. |
| **D básico** | Nodo cabeza, detector, `look_at`, saludo. | El robot detecta a una persona, la mira, la saluda y ejecuta una orden de mano. |
| **C — Brazo** | Inventario, corriente, calibración, PID, URDF. | Rampas con error medido; `fault` por bloqueo; poses nombradas por voz. |
| **D completo** | Distancia estéreo, modo cara a cara sin wake word, saludo con brazo. | Demo integrada de los 5 comportamientos. |
| **Tesis** | Set de evaluación, prueba con 5–10 personas, informe. | Métricas de la sección 10 documentadas. |

Orden: **0 → A → B → D básico → C → D completo → Tesis**. A y B pueden
avanzar en paralelo cuando el servidor esté encendido (A en el Xeon, B en el
banco con el Pico). C va al final del bloque técnico porque es el hardware de
más riesgo y el demo de "interactuar y recibir órdenes" ya existe sin él.

Cada hito produce una entrada en `docs/mediciones/` con fecha, hardware y
números. Eso es directamente material de la tesis.

---

## 10. Métricas para la tesis

| Métrica | Cómo se mide | Meta |
|---|---|---|
| Latencia fin de habla → acción | marcas de `/coramo/state`, 30 órdenes | ≤ 1,5 s (p95) |
| Latencia fin de habla → primer audio | ídem | ≤ 1,8 s (p95) |
| WER del STT en español | set de 50–100 órdenes, 3+ voces, con y sin ruido | ≤ 10 % sin ruido |
| Acierto de tool y argumentos | mismo set | ≥ 90 % |
| Falsas activaciones | 1 h de audio ambiente sin órdenes | 0 acciones físicas |
| Error articular del brazo | rampas por eje, RMS y sobrepaso | definido tras el inventario |
| Tiempo de reacción del watchdog | desconexión USB instrumentada | ≤ 250 ms |
| Detección de personas | FPS y latencia; precisión sobre 100 cuadros anotados | 15 FPS, < 100 ms |
| Prueba con usuarios | 5–10 personas, 3 tareas cada una, cuestionario | reportada, sin meta numérica |
| Costo por orden (backends en nube) | tokens y precio de la API sobre 30 órdenes | reportado |
| Funcionamiento sin internet | desconectar la red, 10 órdenes con respaldo local | 100 % ejecutadas; latencia reportada |

---

## 11. Documentación y gobernanza

- `docs/` es la única fuente de verdad. Estructura: `arquitectura.md` (resumen
  vivo de este spec), `protocolo.md`, `hardware/` (inventario, esquemas,
  alimentación), `instalacion/` (Xeon, RPi5, Pico), `mediciones/` (bitácora
  con fecha), `decisiones/` (registro de decisiones, una por archivo).
- Nombre oficial: **CO**laborativo **R**eprogramable **A**utónomo **MO**dular.
  Se corrige en todo material que diga "Reconfigurable".
- coramo.cl, el paper y la tesis se derivan de `docs/`. El sitio solo se
  actualiza cuando hay un hito medido; no se anuncian predicciones.
- coramo.cl hoy dice "Sin internet. Sin nube.". Cuando v2 use backends en
  nube, el sitio y el paper se corrigen: el mensaje pasa a ser "rápido, con
  respaldo local sin internet".
- Git: rama `v1-rpi5` + tag `v1.0` conservan v1 íntegro. `main` reinicia con
  historia limpia. Los `docs/01..06` de v1 se mueven a `docs/legado/` para
  citarlos como trabajo previo en la tesis.
- Cada subproyecto tiene su spec en `docs/superpowers/specs/` y su plan en
  `docs/superpowers/plans/`, escritos antes de codificar.

---

## 12. Riesgos y decisiones abiertas

| Riesgo / decisión | Mitigación o cuándo se decide |
|---|---|
| El servidor está apagado; nada de la tarea cero se puede verificar hoy | La tarea cero es la primera acción cuando Felipe lo encienda. Este spec asume que Ubuntu está instalado y hay SSH; si no, la tarea cero lo instala. |
| Xeon sin AVX2 | Ninguna inferencia en CPU. Verificar en tarea cero que PyTorch y CTranslate2 importan sin "Illegal instruction". |
| Fuente insuficiente para dos GPUs | Verificar potencia y conectores PCIe antes de encender ambas. |
| VRAM justa (≈ 9 de 12 GB) | Medir en tarea cero; bajar el LLM a Q4_K_M si supera 11 GB. |
| DDS entre dos máquinas | Discovery Server + Ethernet directo + verificación de 10 min. Si aun así falla, respaldo: `rmw_zenoh_cpp`. |
| Cámaras CSI en Ubuntu 24.04 para RPi5 | Verificar con `cam -l` antes de instalar ROS. Si libcamera no las ve, respaldo: Raspberry Pi OS con ROS en contenedor. |
| PWM a 1,5 kHz por PCA9685 | Aceptado. Medir ruido y rizado en C; reabrir con datos si el control de corriente no alcanza. |
| Corriente de bloqueo de los motores vs BTS7960 y fuente | Se conoce en el inventario. Umbrales de `overcurrent` salen de ahí. |
| Estéreo vs RGB-D para distancia | Se decide en D completo con datos de D básico (si la altura de la caja basta para "< 2 m", no hace falta estéreo). |
| Hold por PID ante pérdida de latido consume corriente indefinidamente | Tope de tiempo en `hold` (p. ej. 60 s) tras el cual el brazo baja a una pose de reposo controlada y luego libera. Se define en B. |
| Dependencia de red en demos y ferias | Respaldo local automático; hotspot del teléfono como segunda red; la demo de la tesis se ensaya en ambos modos. |
| Costo acumulado de API | Cache del system prompt; costo por orden medido en la tarea cero; tope de gasto mensual en la cuenta. |
| Privacidad del audio en la nube | Solo se envía el segmento del turno, nunca audio continuo ni video; se declara en la tesis. |
| Modelos que no admiten `tool_choice` forzado (familia Claude Fable) | Usar Sonnet 5 o Haiku 4.5, o `auto` con instrucción explícita y validación del JSON antes de ejecutar. |

---

## 13. Próximos pasos

1. Felipe revisa este documento y lo aprueba o pide cambios.
2. Se escribe el spec + plan del **hito 0** (tarea cero), para ejecutarlo el día
   que se encienda el servidor.
3. Se escribe el spec + plan del **subproyecto A**, que se puede desarrollar
   con `dev-sin-robot.yaml` antes de tener el servidor: los tests del núcleo no
   necesitan GPU (STT y LLM se simulan con salidas grabadas). Las mediciones de
   latencia sí esperan al servidor.
4. B, C y D se especifican a medida que se llega a ellos, con este documento
   como referencia.
