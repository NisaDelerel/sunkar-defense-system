# camera_manager.py
import cv2
from object_detection import detect_objects

class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.frame = None
        self.tracks = []

    def start(self):
        if not self.cap.isOpened():
            print("Kamera açılamadı.")
            return
        self.running = True
        print("Kamera başlatıldı.")

    def stop(self):
        self.running = False
        self.cap.release()
        print("Kamera kapatıldı.")

    def get_frame(self):
        if not self.running:
            print("get_frame çağrıldı ama kamera çalışmıyor.")
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        print("Kare başarıyla alındı.")
        processed_frame, tracks = detect_objects(frame)
        self.frame = processed_frame
        self.tracks = tracks
        return self.frame, self.tracks
