# camera_manager.py
import cv2
import threading
from object_detection import detect_objects
import time

class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.frame = None
        self.tracks = []
        self.lock = threading.Lock()

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

    def get_frame(self):
        with self.lock:
            return self.frame, self.tracks
