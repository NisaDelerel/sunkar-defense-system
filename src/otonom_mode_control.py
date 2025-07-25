# src/otonom_mode_control.py

import cv2
from object_detection import detect_objects
from laser_control import LaserControl

class OtonomModeControl:
    def __init__(self, serial_comm):
        self.serial = serial_comm
        self.laser = LaserControl(self.serial)

    def draw_crosshair(self, frame, bbox, color=(0, 255, 0), size=20, thickness=2):
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.drawMarker(frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)

    def start(self):
        print("[OtonomModeControl] Otonom mod başlatıldı.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[OtonomModeControl] Kamera açılamadı.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[OtonomModeControl] Görüntü alınamadı.")
                break

            # Detect and classify all objects
            _, detections = detect_objects(frame)

            # Filter for red balloons
            red_balloons = [det for det in detections if det['label'].lower() == "red"]

            if not red_balloons:
                print("[OtonomModeControl] Kırmızı balon kalmadı. Otonom mod bitiyor.")
                break

            # Select the red balloon with the smallest track_id
            target = min(red_balloons, key=lambda d: d['track_id'])

            # Draw crosshair
            self.draw_crosshair(frame, target['bbox'])

            # Eliminate the balloon (fire laser)
            print(f"[OtonomModeControl] Hedeflenen balon: track_id={target['track_id']}, bbox={target['bbox']}")
            self.laser.turn_on()
            cv2.waitKey(500)  # Wait for 0.5 seconds to simulate firing
            self.laser.turn_off()

            # Optionally, show the frame for debugging
            cv2.imshow("Otonom Mod", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[OtonomModeControl] Kullanıcı tarafından çıkıldı.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[OtonomModeControl] Otonom mod sonlandı.")