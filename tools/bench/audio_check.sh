#!/bin/bash
# Verificación de audio del Xeon: niveles del micrófono y prueba acústica parlante→micrófono.
# Uso (en el Xeon, con la sesión gráfica de coramo activa): bash tools/bench/audio_check.sh
# Salida esperada: ambiente entre -55 y -40 dBFS, y el tono de 440 Hz al menos 20 dB sobre el fondo.
set -e
export XDG_RUNTIME_DIR=/run/user/1000
python3 - <<'PY'
import wave, math, struct
fr=16000; f=wave.open('/tmp/tono.wav','wb'); f.setnchannels(1); f.setsampwidth(2); f.setframerate(fr)
f.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(2*math.pi*440*i/fr)*min(1,i/800,(fr-i)/800))) for i in range(fr))); f.close()
PY
echo "fuente: $(wpctl get-volume @DEFAULT_AUDIO_SOURCE@)  salida: $(wpctl get-volume @DEFAULT_AUDIO_SINK@)"
amixer -c 0 sget 'Rear Mic Boost' | grep -oE 'Front Left: [0-9]+ \[[0-9]+%\] \[[0-9.]+dB\]' | sed 's/^/boost: /'
arecord -q -D default -d 3 -f S16_LE -r 16000 -c 1 /tmp/amb.wav
(arecord -q -D default -d 5 -f S16_LE -r 16000 -c 1 /tmp/loop.wav &); sleep 1.5; aplay -q -D default /tmp/tono.wav; sleep 3
python3 - <<'PY'
import wave, struct, math
def leer(p):
    w=wave.open(p); d=w.readframes(w.getnframes()); w.close(); return struct.unpack('<%dh'%(len(d)//2), d)
db=lambda v: 20*math.log10(max(v,1e-9)/32768)
def rms(s): return math.sqrt(sum(x*x for x in s)/len(s))
def goertzel(s, f, fr=16000):
    k=int(0.5+len(s)*f/fr); w=2*math.pi*k/len(s); c=2*math.cos(w); s1=s2=0.0
    for x in s: s0=x+c*s1-s2; s2=s1; s1=s0
    return math.sqrt(max(s1*s1+s2*s2-c*s1*s2,0))/(len(s)/2)
amb=leer('/tmp/amb.wav'); print(f"ambiente: RMS {db(rms(amb)):.1f} dBFS | pico {db(max(abs(x) for x in amb)):.1f} dBFS | hum 50 Hz {db(goertzel(amb,50)):.1f} dBFS")
lp=leer('/tmp/loop.wav'); fr=16000; niv=[db(goertzel(lp[i*fr//2:(i+1)*fr//2],440)) for i in range(10)]
base=sum(sorted(niv)[:4])/4; tono=max(niv)
print(f"tono 440 Hz en el mic: {tono:.1f} dBFS vs fondo {base:.1f} dBFS -> {tono-base:.1f} dB ({'OK' if tono-base>20 else 'FALLO: el mic no oye el parlante'})")
PY
