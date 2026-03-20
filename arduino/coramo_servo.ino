#include <Servo.h>
#include <ArduinoJson.h>

Servo servo;
const int SERVO_PIN = 12;

void setup() {
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
  servo.write(90);  // posicion inicial: centro
}

void loop() {
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
      int angulo = doc["servo"];
      angulo = constrain(angulo, 0, 180);
      servo.write(angulo);
      Serial.print("{\"ok\":true,\"angulo\":");
      Serial.print(angulo);
      Serial.println("}");
    } else {
      Serial.println("{\"error\":\"campo servo no encontrado\"}");
    }
  }
}
