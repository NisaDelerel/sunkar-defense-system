# camera_manager.py
import cv2
import threading
from object_detection import detect_objects
import time

class CameraManager:
    def __init__(self, serial_comm= None):
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.frame = None
        self.tracks = []
        self.lock = threading.Lock()
        self.serial = serial_comm

    def start(self):
        if not self.cap.isOpened():
            print("Kamera açılamadı.")
            return
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        print("Kamera başlatıldı.")

    def stop(self):
        self.running = False
        self.cap.release()
        print("Kamera kapatıldı.")

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            processed_frame, tracks = detect_objects(frame)

            with self.lock:
                self.frame = processed_frame
                self.tracks = tracks

            # İstersen buraya sleep koyabilirsin örn. 30 FPS sınırı için:
            time.sleep(0.03)
    def reset_position(self):
        print("Kamera başlangıç konumuna getiriliyor...")
        if self.serial:
            self.serial.send_command(0x01, 90)
            self.serial.send_command(0x02, 0)
            print("Servo 90°, Stepper sıfırlandı.")
        else:
            print("SerialComm bağlı değil, sıfırlama yapılamadı.")

    def move_camera(self, x, y):
        if self.serial:
            servo_angle = int((x + 1) * 90)  # -1..1 -> 0..180 derece örnek
            stepper_steps = int((y + 1) * 100)  # -1..1 -> 0..200 step örnek

            # Komut gönder (komut kodları senin Arduino koduna uyacak şekilde ayarla)
            self.serial.send_command(0x01, servo_angle)
            self.serial.send_command(0x02, stepper_steps)

            print(f"[CameraManager] Servo: {servo_angle}°, Stepper: {stepper_steps} steps")
        else:
            print("[CameraManager] SerialComm bağlı değil, hareket gönderilemedi.")

    def get_frame(self):
        with self.lock:
            return self.frame, self.tracks