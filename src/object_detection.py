# object_detection.py
import os
from ultralytics import YOLO
import numpy as np
import cv2
import joblib
try:
    from tracker.byte_tracker import BYTETracker
except ImportError:
    from tracker.simple_byte_tracker import SimpleBYTETracker as BYTETracker
from tracker_config import BalloonTrackingConfig

# 🔔 Dinamik dosya yolları: Bu dosyanın (object_detection.py) olduğu dizine göre path belirle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
SVM_PATH = os.path.join(BASE_DIR, 'svm_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# ✅ Model ve encoder yükle
model = YOLO(MODEL_PATH)
svm_model = joblib.load(SVM_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Load configuration and create tracker with optimized parameters for balloon tracking
config = BalloonTrackingConfig()
tracker = BYTETracker(
    iou_thresh=config.IOU_THRESH,
    max_missed=config.MAX_MISSED,
    conf_thresh=config.TRACKER_CONF_THRESH
)

def detect_objects(frame):
    # YOLO tahmini with confidence threshold from config
    results = model.predict(source=frame, conf=config.YOLO_CONF_THRESH, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Filter out very small detections which are likely noise
            if w > config.MIN_DETECTION_SIZE and h > config.MIN_DETECTION_SIZE:
                detections.append([x1, y1, w, h, conf, cls_id])

    # BYTETracker ile update - now passing frame for appearance features
    tracks = tracker.update(np.array(detections), frame)

    for track in tracks:
        x, y, w, h = track.tlwh
        track_id = track.track_id

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        roi = frame[y1:y2, x1:x2]

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # ROI'den renk feature çıkar
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_r = np.mean(roi_rgb[:, :, 0])
        avg_g = np.mean(roi_rgb[:, :, 1])
        avg_b = np.mean(roi_rgb[:, :, 2])

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(hsv[:, :, 0]) / 180.0
        avg_s = np.mean(hsv[:, :, 1]) / 255.0
        avg_v = np.mean(hsv[:, :, 2]) / 255.0

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1]) - 128
        avg_b_lab = np.mean(lab[:, :, 2]) - 128

        feature_vector = np.array([[avg_r, avg_g, avg_b, avg_h, avg_s, avg_v, avg_l, avg_a, avg_b_lab]])

        pred_class = svm_model.predict(feature_vector)
        pred_label = label_encoder.inverse_transform(pred_class)[0].strip()

        # Improved visualization with track confidence using config colors
        if track.confidence > config.HIGH_CONF_THRESH:
            color = config.COLOR_HIGH_CONF
        elif track.confidence > config.MED_CONF_THRESH:
            color = config.COLOR_MED_CONF
        else:
            color = config.COLOR_LOW_CONF
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Enhanced text display with confidence
        text = f"ID:{track_id} - {pred_label} ({track.confidence:.2f})"
        
        # Add text background for better visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, max(0, y1 - text_size[1] - 10)), 
                     (x1 + text_size[0], y1), color, -1)
        
        cv2.putText(frame, text, (x1, max(text_size[1], y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, tracks
