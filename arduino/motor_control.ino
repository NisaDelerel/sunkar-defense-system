#include <Servo.h>

#define START_BYTE 0xAA
#define END_BYTE   0x55

// Pin tanımları
const int servoPin = 9;
const int stepPin = 3;
const int dirPin = 4;
const int laserPin = 5;
const int stopPin = 7;

Servo myServo;

// Stepper hız ayarı değişkeni
unsigned int stepperDelay = 800;  // default 800 µs (~625 Hz step rate)

void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);

  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(laserPin, OUTPUT);
  pinMode(stopPin, INPUT_PULLUP);

  digitalWrite(laserPin, LOW);
  myServo.write(90);
}

void loop() {
  if (digitalRead(stopPin) == LOW) {
    emergencyStop();
    while (digitalRead(stopPin) == LOW) delay(10);
  }

  while (Serial.available() >= 4) {
    if (Serial.read() != START_BYTE) continue;
    byte cmd = Serial.read();
    byte data = Serial.read();
    if (Serial.read() != END_BYTE) continue;
    processCommand(cmd, data);
  }
}

void processCommand(byte cmd, byte data) {
  switch (cmd) {
    case 0x01:
      data = constrain(data, 0, 180);
      myServo.write(data);
      Serial.print("S:"); Serial.println(data);
      break;

    case 0x02:
      moveStepper((int8_t)data);
      Serial.print("St:"); Serial.println((int8_t)data);
      break;

    case 0x03:
      // Step motor hız ayarı: 0 (yavaş) – 255 (hızlı)
      data = constrain(data, 1, 255);  // 0 olmaz (bölünemez)
      stepperDelay = map(data, 1, 255, 2000, 200);  // hız aralığı ters orantılı
      Serial.print("Spd:"); Serial.println(data);
      break;

    case 0x04:
      analogWrite(laserPin, data);
      Serial.print("L:"); Serial.println(data);
      break;

    case 0x05:
      {
        int servoPos = myServo.read();
        Serial.print("S:"); Serial.print(servoPos);
        Serial.print(";L:"); Serial.print(analogRead(laserPin));
        Serial.print(";Spd:"); Serial.println(stepperDelay);
      }
      break;

    default:
      Serial.println("ERR:Unknown cmd");
      break;
  }
}

void moveStepper(int8_t steps) {
  digitalWrite(dirPin, steps >= 0 ? HIGH : LOW);
  uint16_t n = abs(steps);
  for (uint16_t i = 0; i < n; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepperDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepperDelay);
  }
}

void emergencyStop() {
  analogWrite(laserPin, 0);
  myServo.detach();
  Serial.println("EMERGENCY STOP");
}
