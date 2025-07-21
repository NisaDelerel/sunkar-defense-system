import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
except ImportError:
    # Fallback for environments without scipy
    def linear_sum_assignment(cost_matrix):
        """Simple greedy assignment fallback"""
        assignments = []
        cost_matrix = np.array(cost_matrix)
        used_rows, used_cols = set(), set()
        
        # Find minimum cost assignments greedily
        while len(assignments) < min(cost_matrix.shape):
            min_cost = float('inf')
            min_pos = None
            
            for i in range(cost_matrix.shape[0]):
                if i in used_rows:
                    continue
                for j in range(cost_matrix.shape[1]):
                    if j in used_cols:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_pos = (i, j)
            
            if min_pos and min_cost < 1.0:
                assignments.append(min_pos)
                used_rows.add(min_pos[0])
                used_cols.add(min_pos[1])
            else:
                break
        
        if assignments:
            return zip(*assignments)
        return [], []
    
    def cdist(a, b, metric='euclidean'):
        """Simple distance calculation fallback"""
        distances = []
        for x in a:
            row = []
            for y in b:
                if metric == 'euclidean':
                    dist = sum((xi - yi)**2 for xi, yi in zip(x, y))**0.5
                else:
                    dist = sum(abs(xi - yi) for xi, yi in zip(x, y))
                row.append(dist)
            distances.append(row)
        return distances

import cv2

class KalmanFilter:
    """Simple Kalman filter for 2D position and velocity tracking"""
    def __init__(self, initial_state):
        # State: [x, y, vx, vy]
        self.state = np.array([initial_state[0], initial_state[1], 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 1.0
        
        # Initial covariance
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        """Update with measurement"""
        z = np.array(measurement, dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class Track:
    def __init__(self, track_id, bbox, class_id, confidence):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.missed = 0
        self.state = 'tentative'  # 'tentative', 'confirmed', 'deleted'
        self.max_age = 30
        self.n_init = 3  # Number of consecutive detections before track is confirmed
        
        # Initialize Kalman filter with center position
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        self.kf = KalmanFilter([cx, cy])
        
        # Store appearance features for matching
        self.appearance_features = []
        self.max_appearance_features = 10
        
    def predict(self):
        """Predict next position using Kalman filter"""
        predicted_pos = self.kf.predict()
        # Update bbox with predicted center position
        cx, cy = predicted_pos[0], predicted_pos[1]
        x = cx - self.bbox[2]/2
        y = cy - self.bbox[3]/2
        self.predicted_bbox = (x, y, self.bbox[2], self.bbox[3])
        return self.predicted_bbox
    
    def update(self, bbox, confidence, appearance_feature=None):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.missed = 0
        self.age += 1
        
        # Update Kalman filter
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        self.kf.update([cx, cy])
        
        # Update appearance features
        if appearance_feature is not None:
            self.appearance_features.append(appearance_feature)
            if len(self.appearance_features) > self.max_appearance_features:
                self.appearance_features.pop(0)
        
        # Update track state
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed"""
        self.missed += 1
        self.age += 1
        
        if self.state == 'tentative':
            # Delete tentative tracks quickly
            if self.missed > 1:
                self.state = 'deleted'
        elif self.state == 'confirmed':
            # Keep confirmed tracks longer
            if self.missed > self.max_age:
                self.state = 'deleted'
    
    def is_confirmed(self):
        return self.state == 'confirmed'
    
    def is_deleted(self):
        return self.state == 'deleted'
    
    @property
    def tlwh(self):
        return self.bbox
    
    def get_average_appearance(self):
        """Get average appearance feature"""
        if not self.appearance_features:
            return None
        return np.mean(self.appearance_features, axis=0)

class BYTETracker:
    def __init__(self, iou_thresh=0.7, max_missed=30, conf_thresh=0.6):
        self.next_id = 1
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.conf_thresh = conf_thresh
        self.appearance_weight = 0.3  # Weight for appearance similarity

    def update(self, detections, frame=None):
        """Update tracker with new detections"""
        # Filter detections by confidence
        detections = [d for d in detections if d[4] >= self.conf_thresh]
        
        if len(detections) == 0:
            # No detections, mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            # Remove deleted tracks
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
            return [t for t in self.tracks if t.is_confirmed()]
        
        # Predict track positions
        for track in self.tracks:
            track.predict()
        
        # Extract appearance features if frame is provided
        appearance_features = []
        if frame is not None:
            appearance_features = self._extract_appearance_features(detections, frame)
        
        # Separate confirmed and tentative tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        tentative_tracks = [t for t in self.tracks if t.state == 'tentative']
        
        # First association: confirmed tracks with high confidence detections
        high_conf_dets = [i for i, d in enumerate(detections) if d[4] >= 0.7]
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            confirmed_tracks, detections, high_conf_dets, appearance_features
        )
        
        # Second association: remaining tracks with remaining detections
        remaining_dets = [i for i in range(len(detections)) if i not in unmatched_dets]
        low_conf_dets = [i for i in remaining_dets if i not in high_conf_dets]
        
        if len(unmatched_tracks) > 0 and len(low_conf_dets) > 0:
            matched_tracks_2, unmatched_tracks_2, unmatched_dets_2 = self._associate(
                unmatched_tracks, detections, low_conf_dets, appearance_features
            )
            matched_tracks.extend(matched_tracks_2)
            unmatched_tracks = unmatched_tracks_2
            unmatched_dets.extend(unmatched_dets_2)
        
        # Third association: tentative tracks with remaining detections
        remaining_dets = [i for i in range(len(detections)) if i not in unmatched_dets]
        if len(tentative_tracks) > 0 and len(remaining_dets) > 0:
            matched_tracks_3, unmatched_tentative, unmatched_dets_3 = self._associate(
                tentative_tracks, detections, remaining_dets, appearance_features
            )
            matched_tracks.extend(matched_tracks_3)
            unmatched_tracks.extend(unmatched_tentative)
            unmatched_dets.extend(unmatched_dets_3)
        else:
            unmatched_tracks.extend(tentative_tracks)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx] if track_idx < len(self.tracks) else None
            if track is not None:
                x, y, w, h, conf, cls = detections[det_idx]
                appearance_feat = appearance_features[det_idx] if appearance_features else None
                track.update((x, y, w, h), conf, appearance_feat)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):
                self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        remaining_dets = [i for i in range(len(detections)) if i not in [m[1] for m in matched_tracks]]
        for det_idx in remaining_dets:
            x, y, w, h, conf, cls = detections[det_idx]
            appearance_feat = appearance_features[det_idx] if appearance_features else None
            new_track = Track(self.next_id, (x, y, w, h), cls, conf)
            if appearance_feat is not None:
                new_track.appearance_features.append(appearance_feat)
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _associate(self, tracks, detections, detection_indices, appearance_features):
        """Associate tracks with detections using Hungarian algorithm"""
        if len(tracks) == 0 or len(detection_indices) == 0:
            return [], list(range(len(tracks))), detection_indices
        
        # Create cost matrix
        cost_matrix = np.zeros((len(tracks), len(detection_indices)))
        
        for i, track in enumerate(tracks):
            for j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                x, y, w, h, conf, cls = det
                
                # IoU cost
                iou = self._iou(track.predicted_bbox, (x, y, w, h))
                iou_cost = 1.0 - iou
                
                # Appearance cost
                appearance_cost = 0.0
                if appearance_features and track.get_average_appearance() is not None:
                    appearance_sim = self._appearance_similarity(
                        track.get_average_appearance(), appearance_features[det_idx]
                    )
                    appearance_cost = 1.0 - appearance_sim
                
                # Combined cost
                total_cost = (1.0 - self.appearance_weight) * iou_cost + self.appearance_weight * appearance_cost
                
                # Set high cost for low IoU
                if iou < self.iou_thresh:
                    total_cost = 1.0
                
                cost_matrix[i, j] = total_cost
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter valid assignments
        matched_tracks = []
        unmatched_tracks = []
        unmatched_dets = []
        
        for i, track in enumerate(tracks):
            if i in row_indices:
                j = col_indices[np.where(row_indices == i)[0][0]]
                if cost_matrix[i, j] < 1.0:  # Valid assignment
                    matched_tracks.append((i, detection_indices[j]))
                else:
                    unmatched_tracks.append(i)
            else:
                unmatched_tracks.append(i)
        
        for j, det_idx in enumerate(detection_indices):
            if j not in col_indices or cost_matrix[row_indices[np.where(col_indices == j)[0][0]], j] >= 1.0:
                unmatched_dets.append(det_idx)
        
        return matched_tracks, unmatched_tracks, unmatched_dets
    
    def _extract_appearance_features(self, detections, frame):
        """Extract simple appearance features from detections"""
        features = []
        
        for det in detections:
            x, y, w, h, conf, cls = det
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(6))  # Default feature if invalid ROI
                continue
            
            roi = frame[y1:y2, x1:x2]
            
            # Extract color histogram features
            hist_b = cv2.calcHist([roi], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([roi], [2], None, [16], [0, 256])
            
            # Normalize histograms
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)
            
            # Combine features
            feature = np.concatenate([hist_b, hist_g, hist_r])
            features.append(feature)
        
        return features
    
    def _appearance_similarity(self, feat1, feat2):
        """Calculate appearance similarity using cosine similarity"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def _iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU)"""
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
