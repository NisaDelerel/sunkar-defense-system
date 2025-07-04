# object_detection.py
from ultralytics import YOLO
import numpy as np
import cv2
from tracker.byte_tracker import BYTETracker

model = YOLO('yolov8m.pt')
tracker = BYTETracker()

def detect_objects(frame):
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append([x1, y1, w, h, conf, cls_id])

    tracks = tracker.update(np.array(detections), frame)

    for track in tracks:
        x, y, w, h = track.tlwh
        track_id = track.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame, tracks
