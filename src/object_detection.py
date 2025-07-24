import os
import sys
from ultralytics import YOLO
import numpy as np
import cv2
import joblib

# --- BYTETrack path setup ---
BYTE_TRACK_PATH = os.path.join(os.path.dirname(__file__), "..", "ByteTrack")
if BYTE_TRACK_PATH not in sys.path:
    sys.path.append(BYTE_TRACK_PATH)

from yolox.tracker.byte_tracker import BYTETracker

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
SVM_PATH = os.path.join(BASE_DIR, 'svm_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# --- Models ---
model = YOLO(MODEL_PATH)
svm_model = joblib.load(SVM_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# --- BYTETracker initialization ---
import types
args = types.SimpleNamespace()
args.track_thresh = 0.5
args.track_buffer = 30
args.match_thresh = 0.8
args.mot20 = False  # Set True if using MOT20 dataset, else False

tracker = BYTETracker(args, frame_rate=30)  # Set frame_rate to your camera's FPS

def detect_objects(frame):
    results = model.predict(source=frame, conf=0.7, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append([x1, y1, x2, y2, conf])

    dets_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)

    # BYTETrack update expects (detections, img_info, img_size)
    online_targets = tracker.update(dets_np, frame.shape[:2], frame.shape[:2])
    detection_dicts = []
    for t in online_targets:
        tlwh = t.tlwh  # top-left x, y, width, height
        track_id = t.track_id
        x, y, w, h = tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # RGB ortalamaları
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_r = np.mean(roi_rgb[:, :, 0])
        avg_g = np.mean(roi_rgb[:, :, 1])
        avg_b = np.mean(roi_rgb[:, :, 2])

        # HSV ortalamaları
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(hsv[:, :, 0]) / 180.0
        avg_s = np.mean(hsv[:, :, 1]) / 255.0
        avg_v = np.mean(hsv[:, :, 2]) / 255.0

        # LAB ortalamaları
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1]) - 128
        avg_b_lab = np.mean(lab[:, :, 2]) - 128

        # Özellik vektörü oluştur
        feature_vector = np.array([[avg_r, avg_g, avg_b, avg_h, avg_s, avg_v, avg_l, avg_a, avg_b_lab]])

        # SVM ile sınıf tahmini
        pred_class = svm_model.predict(feature_vector)
        pred_label = label_encoder.inverse_transform(pred_class)[0].strip()

        detection_dicts.append({
            'track_id': track_id,
            'bbox': (x1, y1, x2, y2),
            'label': pred_label,
            'confidence': t.score if hasattr(t, 'score') else None
        })

    return frame, detection_dicts
