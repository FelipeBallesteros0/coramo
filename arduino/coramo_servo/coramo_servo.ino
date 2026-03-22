#include <Servo.h>
#include <ArduinoJson.h>

Servo servo;
const int SERVO_PIN = 12;

// Estado de oscilacion
bool oscilando = false;
int oscMin = 0, oscMax = 180, oscVel = 15;
int oscPos = 0, oscDir = 1;

void setup() {
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
  servo.write(90);  // posicion inicial: centro
}

void loop() {
  // Modo oscilacion continua (no bloqueante)
  if (oscilando) {
    // Verificar si llego comando detener
    if (Serial.available() > 0) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      JsonDocument doc;
      if (!deserializeJson(doc, line) && doc["detener"].is<bool>()) {
        oscilando = false;
        Serial.println("{\"ok\":true,\"estado\":\"detenido\"}");
        return;
      }
    }
    // Mover un paso
    oscPos += oscDir;
    if (oscPos >= oscMax) { oscPos = oscMax; oscDir = -1; }
    if (oscPos <= oscMin) { oscPos = oscMin; oscDir =  1; }
    servo.write(oscPos);
    delay(oscVel);
    return;
  }

  // Modo normal: esperar comando
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, line);
    if (err) {
      Serial.println("{\"error\":\"json invalido\"}");
      return;
    }

    if (doc["servo"].is<int>()) {
      // Mover a angulo fijo
      int angulo = constrain((int)doc["servo"], 0, 180);
      servo.write(angulo);
      Serial.print("{\"ok\":true,\"angulo\":");
      Serial.print(angulo);
      Serial.println("}");

    } else if (doc["barrer"].is<JsonObject>()) {
      // Barrido ida+vuelta N veces
      JsonObject b = doc["barrer"];
      int ini = constrain((int)(b["inicio"] | 0),   0, 180);
      int fin = constrain((int)(b["fin"]    | 180),  0, 180);
      int rps = max(1, (int)(b["reps"] | 1));
      int vel = max(1, (int)(b["vel"]  | 15));
      for (int r = 0; r < rps; r++) {
        if (ini <= fin) {
          for (int a = ini; a <= fin; a++) { servo.write(a); delay(vel); }
          for (int a = fin; a >= ini; a--) { servo.write(a); delay(vel); }
        } else {
          for (int a = ini; a >= fin; a--) { servo.write(a); delay(vel); }
          for (int a = fin; a <= ini; a++) { servo.write(a); delay(vel); }
        }
      }
      Serial.println("{\"ok\":true,\"accion\":\"barrido\"}");

    } else if (doc["oscilar"].is<JsonObject>()) {
      // Iniciar oscilacion continua
      JsonObject o = doc["oscilar"];
      oscMin = constrain((int)(o["min"] | 0),   0, 180);
      oscMax = constrain((int)(o["max"] | 180),  0, 180);
      oscVel = max(1, (int)(o["vel"] | 15));
      oscPos = oscMin;
      oscDir = 1;
      oscilando = true;
      Serial.println("{\"ok\":true,\"estado\":\"oscilando\"}");

    } else if (doc["detener"].is<bool>()) {
      // Detener cualquier movimiento
      oscilando = false;
      Serial.println("{\"ok\":true,\"estado\":\"detenido\"}");

    } else {
      Serial.println("{\"error\":\"comando desconocido\"}");
    }
  }
}
