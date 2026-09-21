# Subproyecto A: mediciones (2026-09-20)

Hardware: Xeon E5-2697 v2, RTX 4070 SUPER, Ubuntu 26.04, ROS 2 Lyrical Luth. Cuerpo simulado. Entrada: las 30 órdenes grabadas del hito 0 (`~/datos/ordenes/`), reproducidas a ritmo real por el servidor de habla. Herramienta: `tools/medir_subproyecto_a.py`.

## Cómo repetir estas mediciones

```bash
./tools/correr_aceptacion.sh latencias      # las 30 órdenes grabadas
./tools/correr_aceptacion.sh autoescucha    # veinte respuestas largas al aire
./tools/correr_aceptacion.sh recuperacion   # tumba cada servidor por turno
```

## Latencias

Todo se correlaciona por `speech_end`, el instante en que el usuario deja de hablar, que viaja dentro de los mensajes.

| Tramo | n | p50 | p95 | Meta | Estado |
|---|---|---|---|---|---|
| Cierre del turno (`speech_end` → transcripción) | 30 | 0,75 s | 0,79 s | 0,80 s | cumple |
| **Total hasta la acción** (`speech_end` → comando validado) | 20 | **1,09 s** | **1,27 s** | 1,50 s | **cumple** |
| Hasta pedir la respuesta hablada | 8 | 1,21 s | 1,52 s | 1,60 s | cumple |

Las **30 órdenes** se transcriben. De ellas, 28 producen decisión: las dos últimas las rechaza el filtro de seguridad porque las cuatro órdenes de parada que las preceden dejan el robot enclavado, y nadie lo rearma. Es el comportamiento diseñado.

De los 0,73 s del cierre del turno, **0,60 son la espera de silencio** que confirma que la persona terminó de hablar, y el resto la transcripción. Es una constante configurable, no tiempo de cómputo.

## Acierto de herramienta

**27 de 28 (96 %)**, por encima del 90 % que pide el spec.

El único fallo es `«Coramo ASRock»`, que es cómo transcribe Whisper la orden «coramo haz rock». Ante ese texto sin sentido el modelo eligió `responder`, que es el comportamiento correcto: no inventar un movimiento. El problema está en la transcripción de anglicismos, ya observado en el hito 0, no en la decisión.

## Auto-escucha

Veinte respuestas largas seguidas por el parlante, con el micrófono real abierto: **cero activaciones y cero transcripciones**. Herramienta: `tools/prueba_autoescucha.py`.

La prueba empieza por un control que la valida: dice una frase **saltándose el silenciado** y exige oírse a sí misma. El micrófono transcribe esa frase entera, y contiene la palabra de activación. Sin silenciar, entonces, el robot **sí** se obedecería a sí mismo: el cero de arriba mide el silenciado, no un micrófono apagado.

## Recuperación ante caídas

Se tumba cada servidor por separado y se comprueba qué sigue en pie. Herramienta: `tools/prueba_recuperacion.py`. Los seis puntos cumplen.

| Se cae | Lo que se exige | Resultado |
|---|---|---|
| Servidor de voz | una orden física llega igual a `/body/command_safe` | `mano`/`cierra` publicado |
| Servidor de voz | una pregunta no tumba ningún nodo | los 7 nodos siguen vivos |
| Modelo de lenguaje | la parada sigue siendo inmediata | `detener` publicado |
| Modelo de lenguaje | una orden normal falla limpio | sin comando, sin caída |
| Servidor de habla | el cerebro sobrevive | los 7 nodos siguen vivos |
| Servidor de habla | el nodo se reconecta solo al volver | transcripción recibida |

La parada sobrevive a la caída del modelo de lenguaje porque no pasa por él: el nodo del agente la reconoce por texto y publica el comando directo.

## Errores encontrados al construir, y lo que enseñaron

Siete fallos reales, ninguno detectado por las pruebas unitarias: todos aparecieron al ejercitar el sistema completo.

| Qué fallaba | Causa | Cómo se vio |
|---|---|---|
| El generador de mensajes de ROS no compilaba | Faltaban `python3-empy` y `python3-lark`, y CMake elegía el Python 3.12 de `uv` en vez del del sistema | `ModuleNotFoundError: No module named 'em'`, aunque `import em` funcionaba en la terminal |
| El nodo de seguridad moría al rechazar un comando | `RcutilsLogger` no tiene `warn()` en esta versión, sino `warning()` | El primer comando fuera de límite tumbaba el nodo |
| El robot parecía hablar sin que sonara nada | Bajo systemd faltaba `XDG_RUNTIME_DIR` y `aplay` no alcanzaba a PipeWire; el error se tragaba | La llamada tardaba 0,25 s en vez de 1,9 s: no había reproducción |
| La primera frase tardaba 1,8 s en vez de 0,13 s | El modelo de voz cargaba al primer uso | Se resolvió calentando al arrancar el servicio |
| Con audio grabado no salía ninguna transcripción | El detector medía el silencio con el reloj de pared, y los archivos se procesaban más rápido que el tiempo real | El turno nunca se cerraba |
| «coramo cierra la mano» se transcribía «Decoramos Sierra La Mano» | El detector cortaba el ataque de la primera palabra | Se guardan 320 ms previos al inicio del habla |
| El cerebro perdía 4 de cada 30 órdenes | El flujo de eventos usaba **una sola cola compartida**: cada evento llegaba a un único suscriptor, así que cualquier visor de depuración le robaba turnos al cerebro | Dos lectores simultáneos recibían listas distintas y complementarias |
| Las latencias salían de 6 a 14 s | El cliente leía bloques de 1024 bytes y esperaba a llenarlos antes de entregar cada evento | Pasó a leer por líneas: de 6,16 s a 0,73 s |

La lección que más se repitió: **medir en la máquina que produce el dato antes de culpar al resto**. El nodo publicaba a 15 Hz perfectos mientras el consumidor veía 1,2 Hz, y el detector de habla iba a tiempo real mientras las latencias parecían de seis segundos.


## Las cuatro órdenes que faltaban

Merecen contarse aparte porque la primera explicación era falsa.

Archivo por archivo, el detector encuentra **un turno limpio en los 30** y Whisper los transcribe bien: ni el audio ni la detección tenían la culpa. Los audios que el propio servidor guarda de cada turno confirmaban 30 turnos cerrados. La pérdida estaba más arriba.

Había dos causas encadenadas.

La primera es un defecto real del servidor, ya corregido: **una sola cola para todos los suscriptores** del flujo de eventos. Cada evento salía hacia un único cliente, de modo que un segundo lector conectado para mirar le quitaba turnos al cerebro sin que nada lo delatara. Ahora cada conexión tiene su propia cola y `/health` informa cuántos suscriptores hay, que es como se detecta un oyente colado.

La segunda no era del robot sino del banco de pruebas. Las órdenes perdidas eran siempre las que sonaban **mientras el robot contestaba**: la 09 tras «haz rock», la 19 tras «qué hora es», la 22 y la 23 tras el chiste. La 24 llegaba partida, «*Amo* buenos días», porque el micrófono se reabría a media palabra. Es el antieco haciendo su trabajo; lo que fallaba es que una grabación le habla encima al robot y una persona no. La reproducción de archivos ahora **espera a que el robot termine** antes de soltar la orden siguiente, y deja 2,6 s entre órdenes para que la decisión alcance a llegar.

Esa espera trajo un tercer problema, de medición: el reloj de audio avanza con las muestras, y durante la pausa no hay muestras, así que se quedaba atrás del reloj de pared y las latencias salían infladas hasta 7 s. Al reanudar hay que **reanclar** el reloj. Con eso, 30 de 30 y las metas se cumplen.
