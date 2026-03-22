#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <ArduinoJson.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

const int NUM_DEDOS = 5;
const int POS0   = 172;   // PWM para 0 grados
const int POS180 = 565;   // PWM para 180 grados
// Pin 0 = pulgar (invertido), pins 1-4 = dedos normales

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setPWMFreq(50);  // servos analogicos: 50 Hz
  // Posicion inicial: mano abierta
  for (int i = 0; i < NUM_DEDOS; i++) mover_dedo(i, 0);
  Serial.println("{\"ok\":true,\"estado\":\"listo\"}");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    JsonDocument doc;
    if (deserializeJson(doc, line)) {
      Serial.println("{\"error\":\"json invalido\"}");
      return;
    }

    if (doc["dedo"].is<int>() && doc["angulo"].is<int>()) {
      int dedo   = constrain((int)doc["dedo"],   0, NUM_DEDOS - 1);
      int angulo = constrain((int)doc["angulo"], 0, 180);
      mover_dedo(dedo, angulo);
      Serial.print("{\"ok\":true,\"dedo\":"); Serial.print(dedo);
      Serial.print(",\"angulo\":"); Serial.print(angulo);
      Serial.println("}");

    } else if (doc["gesto"].is<const char*>()) {
      String gesto = doc["gesto"].as<String>();
      if (gesto == "abre") {
        for (int i = 0; i < NUM_DEDOS; i++) mover_dedo(i, 0);
        Serial.println("{\"ok\":true,\"gesto\":\"abre\"}");
      } else if (gesto == "cierra") {
        for (int i = 0; i < NUM_DEDOS; i++) mover_dedo(i, 180);
        Serial.println("{\"ok\":true,\"gesto\":\"cierra\"}");
      } else {
        Serial.println("{\"error\":\"gesto desconocido\"}");
      }

    } else {
      Serial.println("{\"error\":\"comando desconocido\"}");
    }
  }
}

void mover_dedo(int dedo, int angulo) {
  int pwm;
  if (dedo == 0) {
    // Pulgar invertido: 0 grados -> POS180, 180 grados -> POS0
    pwm = map(angulo, 0, 180, POS180, POS0);
  } else {
    pwm = map(angulo, 0, 180, POS0, POS180);
  }
  pca.setPWM(dedo, 0, pwm);
}
