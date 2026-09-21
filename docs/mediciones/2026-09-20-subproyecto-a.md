# Subproyecto A: mediciones (2026-09-20)

Hardware: Xeon E5-2697 v2, RTX 4070 SUPER, Ubuntu 26.04, ROS 2 Lyrical Luth. Cuerpo simulado. Entrada: las 30 órdenes grabadas del hito 0 (`~/datos/ordenes/`), reproducidas a ritmo real por el servidor de habla. Herramienta: `tools/medir_subproyecto_a.py`.

## Latencias

Todo se correlaciona por `speech_end`, el instante en que el usuario deja de hablar, que viaja dentro de los mensajes.

| Tramo | n | p50 | p95 | Meta | Estado |
|---|---|---|---|---|---|
| Cierre del turno (`speech_end` → transcripción) | 26 | 0,73 s | 0,76 s | 0,80 s | cumple |
| **Total hasta la acción** (`speech_end` → comando validado) | 19 | **1,08 s** | **1,25 s** | 1,50 s | **cumple** |
| Hasta pedir la respuesta hablada | 3 | 1,22 s | 1,49 s | 1,60 s | cumple |

De los 0,73 s del cierre del turno, **0,60 son la espera de silencio** que confirma que la persona terminó de hablar, y el resto la transcripción. Es una constante configurable, no tiempo de cómputo.

## Acierto de herramienta

**21 de 22 (95 %)**, por encima del 90 % que pide el spec.

El único fallo es `«Coramo ASRock»`, que es cómo transcribe Whisper la orden «coramo haz rock». Ante ese texto sin sentido el modelo eligió `responder`, que es el comportamiento correcto: no inventar un movimiento. El problema está en la transcripción de anglicismos, ya observado en el hito 0, no en la decisión.

## Lo que no cierra todavía

- **Se transcribieron 26 de 30 órdenes.** Faltan cuatro, y cuatro transcripciones más no produjeron decisión, casi seguro porque la palabra de activación no sobrevivió a la transcripción. Hay que revisar cuáles, con los audios que el servidor guarda en `~/datos/sesiones/`.
- **Sin verificar aún:** cero auto-escuchas en veinte respuestas largas seguidas, y la recuperación al matar cada servidor por separado. Ambos criterios del spec, §11.

## Errores encontrados al construir, y lo que enseñaron

Seis fallos reales, ninguno detectado por las pruebas unitarias: todos aparecieron al ejercitar el sistema completo.

| Qué fallaba | Causa | Cómo se vio |
|---|---|---|
| El generador de mensajes de ROS no compilaba | Faltaban `python3-empy` y `python3-lark`, y CMake elegía el Python 3.12 de `uv` en vez del del sistema | `ModuleNotFoundError: No module named 'em'`, aunque `import em` funcionaba en la terminal |
| El nodo de seguridad moría al rechazar un comando | `RcutilsLogger` no tiene `warn()` en esta versión, sino `warning()` | El primer comando fuera de límite tumbaba el nodo |
| El robot parecía hablar sin que sonara nada | Bajo systemd faltaba `XDG_RUNTIME_DIR` y `aplay` no alcanzaba a PipeWire; el error se tragaba | La llamada tardaba 0,25 s en vez de 1,9 s: no había reproducción |
| La primera frase tardaba 1,8 s en vez de 0,13 s | El modelo de voz cargaba al primer uso | Se resolvió calentando al arrancar el servicio |
| Con audio grabado no salía ninguna transcripción | El detector medía el silencio con el reloj de pared, y los archivos se procesaban más rápido que el tiempo real | El turno nunca se cerraba |
| «coramo cierra la mano» se transcribía «Decoramos Sierra La Mano» | El detector cortaba el ataque de la primera palabra | Se guardan 320 ms previos al inicio del habla |
| Las latencias salían de 6 a 14 s | El cliente leía bloques de 1024 bytes y esperaba a llenarlos antes de entregar cada evento | Pasó a leer por líneas: de 6,16 s a 0,73 s |

La lección que más se repitió: **medir en la máquina que produce el dato antes de culpar al resto**. El nodo publicaba a 15 Hz perfectos mientras el consumidor veía 1,2 Hz, y el detector de habla iba a tiempo real mientras las latencias parecían de seis segundos.
