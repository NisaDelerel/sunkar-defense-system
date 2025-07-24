#include <Servo.h>

// Pin assignments
const int servoPin = 9;      // FT5335M-FB signal
const int stepPin = 3;       // A4988 STEP
const int dirPin = 4;        // A4988 DIR
const int laserPin = 5;      // Laser enable (HIGH=ON)
const int estopPin = 7;      // Emergency stop button (active LOW)

// Servo (pitch) parameters
const int servoMinPulse = 900;   // µs
const int servoMaxPulse = 2100;  // µs
const int pitchMin = 0;          // deg
const int pitchMax = 60;         // deg

// Stepper (yaw) parameters
const int stepsPerRev = 200;     // NEMA 17, adjust if microstepping
const float yawMin = 0.0;        // deg
const float yawMax = 270.0;      // deg
const float noFireZoneMin = -15.0; // deg
const float noFireZoneMax =  15.0; // deg

// State variables
float currentYaw = 0.0;          // deg
int currentPitch = 0;            // deg
bool firing = false;
bool estop = false;

// Stepper tracking
long currentStep = 0;
const float degPerStep = (yawMax - yawMin) / stepsPerRev; // adjust if microstepping

Servo pitchServo;

void setup() {
  Serial.begin(115200);
  pitchServo.attach(servoPin, servoMinPulse, servoMaxPulse);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(laserPin, OUTPUT);
  pinMode(estopPin, INPUT_PULLUP);
  digitalWrite(laserPin, LOW);
  movePitch(pitchMin); // Initialize at 0°
  moveYaw(yawMin);     // Initialize at 0°
}

void loop() {
  // Emergency stop check
  if (digitalRead(estopPin) == LOW) {
    estop = true;
    stopAll();
    Serial.println("ESTOP");
    while (digitalRead(estopPin) == LOW) delay(10); // Wait for button release
  }

  if (estop) {
    // Wait for reset command
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd == "RESET") {
        estop = false;
        pitchServo.attach(servoPin, servoMinPulse, servoMaxPulse);
        Serial.println("RESET OK");
      }
    }
    return;
  }

  // Serial command handling
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    handleCommand(cmd);
  }
}

void handleCommand(String cmd) {
  if (cmd.startsWith("YAW:")) {
    float targetYaw = cmd.substring(4).toFloat();
    if (targetYaw >= yawMin && targetYaw <= yawMax) {
      moveYaw(targetYaw);
      Serial.print("YAW_OK:");
      Serial.println(targetYaw);
    } else {
      Serial.println("ERR:YAW_RANGE");
    }
  } else if (cmd.startsWith("PITCH:")) {
    int targetPitch = cmd.substring(6).toInt();
    if (targetPitch >= pitchMin && targetPitch <= pitchMax) {
      movePitch(targetPitch);
      Serial.print("PITCH_OK:");
      Serial.println(targetPitch);
    } else {
      Serial.println("ERR:PITCH_RANGE");
    }
  } else if (cmd.startsWith("FIRE:")) {
    int fireCmd = cmd.substring(5).toInt();
    setFiring(fireCmd == 1);
    Serial.print("FIRE_OK:");
    Serial.println(fireCmd);
  } else if (cmd == "ESTOP") {
    estop = true;
    stopAll();
    Serial.println("ESTOP");
  } else if (cmd == "STATUS") {
    sendStatus();
  } else {
    Serial.println("ERR:UNKNOWN_CMD");
  }
}

void moveYaw(float targetYaw) {
  // Convert angle to steps
  long targetStep = (long)((targetYaw - yawMin) / (yawMax - yawMin) * stepsPerRev);
  long delta = targetStep - currentStep;
  if (delta == 0) return;
  digitalWrite(dirPin, delta > 0 ? HIGH : LOW);
  for (long i = 0; i < abs(delta); i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(800); // adjust for speed
    digitalWrite(stepPin, LOW);
    delayMicroseconds(800);
  }
  currentStep = targetStep;
  currentYaw = yawMin + (float)currentStep * (yawMax - yawMin) / stepsPerRev;
}

void movePitch(int targetPitch) {
  int pulse = map(targetPitch, pitchMin, pitchMax, servoMinPulse, servoMaxPulse);
  pitchServo.writeMicroseconds(pulse);
  currentPitch = targetPitch;
}

void setFiring(bool enable) {
  // Only allow firing if not in no-fire zone
  if (currentYaw > noFireZoneMin && currentYaw < noFireZoneMax) {
    digitalWrite(laserPin, LOW); // No fire
    firing = false;
  } else {
    digitalWrite(laserPin, enable ? HIGH : LOW);
    firing = enable;
  }
}

void stopAll() {
  digitalWrite(laserPin, LOW);
  firing = false;
  // Optionally detach servo for safety
  pitchServo.detach();
  // Stepper: no action needed (no holding torque)
}

void sendStatus() {
  Serial.print("YAW:"); Serial.print(currentYaw);
  Serial.print(";PITCH:"); Serial.print(currentPitch);
  Serial.print(";FIRE:"); Serial.print(firing ? 1 : 0);
  Serial.print(";ESTOP:"); Serial.println(estop ? 1 : 0);
}