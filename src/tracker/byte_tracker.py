import numpy as np

class Track:
    def __init__(self, track_id, bbox, class_id):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.class_id = class_id
        self.age = 0
        self.hits = 1
        self.missed = 0

    @property
    def tlwh(self):
        return self.bbox

class BYTETracker:
    def __init__(self, iou_thresh=0.5, max_missed=30):
        self.next_id = 1
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed

    def update(self, detections, frame=None):
        updated_tracks = []
        unmatched_detections = []

        detections = [d for d in detections if d[4] > 0.3]  # confidence filter

        for det in detections:
            x, y, w, h, conf, cls = det
            best_iou = 0
            best_track = None

            for track in self.tracks:
                iou = self.iou(track.bbox, (x, y, w, h))
                if iou > best_iou and iou >= self.iou_thresh:
                    best_iou = iou
                    best_track = track

            if best_track:
                best_track.bbox = (x, y, w, h)
                best_track.class_id = cls
                best_track.hits += 1
                best_track.missed = 0
                updated_tracks.append(best_track)
            else:
                unmatched_detections.append((x, y, w, h, cls))

        for track in self.tracks:
            if track not in updated_tracks:
                track.missed += 1

        for x, y, w, h, cls in unmatched_detections:
            new_track = Track(self.next_id, (x, y, w, h), cls)
            self.next_id += 1
            updated_tracks.append(new_track)

        self.tracks = [t for t in updated_tracks if t.missed <= self.max_missed]
        return self.tracks

    def iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
