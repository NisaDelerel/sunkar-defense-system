"""
Simplified BYTETracker with key improvements for balloon tracking stability.
This version minimizes dependencies while maintaining core improvements:
1. Multi-stage association
2. Track state management  
3. Motion prediction (simplified)
4. Appearance features (basic)
5. Hungarian algorithm (or greedy fallback)
"""

class SimpleKalmanFilter:
    """Simplified Kalman filter using basic motion prediction"""
    def __init__(self, initial_pos):
        self.x, self.y = initial_pos
        self.vx, self.vy = 0.0, 0.0  # velocity
        self.prev_x, self.prev_y = initial_pos
        
    def predict(self):
        """Predict next position based on velocity"""
        # Update velocity based on previous movement
        self.vx = 0.7 * self.vx + 0.3 * (self.x - self.prev_x)
        self.vy = 0.7 * self.vy + 0.3 * (self.y - self.prev_y)
        
        # Predict next position
        predicted_x = self.x + self.vx
        predicted_y = self.y + self.vy
        
        return [predicted_x, predicted_y]
    
    def update(self, measurement):
        """Update with new measurement"""
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = measurement

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
        
        # Initialize simple motion predictor
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        self.motion_filter = SimpleKalmanFilter([cx, cy])
        
        # Store appearance features for matching (simplified)
        self.appearance_features = []
        self.max_appearance_features = 5
        
    def predict(self):
        """Predict next position"""
        predicted_pos = self.motion_filter.predict()
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
        
        # Update motion filter
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        self.motion_filter.update([cx, cy])
        
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
        # Simple average
        if isinstance(self.appearance_features[0], (list, tuple)):
            avg_feature = []
            feature_len = len(self.appearance_features[0])
            for i in range(feature_len):
                avg_val = sum(f[i] for f in self.appearance_features) / len(self.appearance_features)
                avg_feature.append(avg_val)
            return avg_feature
        return self.appearance_features[-1]  # Return latest if not list

class SimpleBYTETracker:
    def __init__(self, iou_thresh=0.5, max_missed=20, conf_thresh=0.5):
        self.next_id = 1
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.conf_thresh = conf_thresh
        self.appearance_weight = 0.3

    def update(self, detections, frame=None):
        """Update tracker with new detections"""
        # Filter detections by confidence
        detections = [d for d in detections if len(d) >= 5 and d[4] >= self.conf_thresh]
        
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
            try:
                appearance_features = self._extract_appearance_features(detections, frame)
            except:
                # If appearance extraction fails, continue without it
                appearance_features = [None] * len(detections)
        
        # Separate confirmed and tentative tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        tentative_tracks = [t for t in self.tracks if t.state == 'tentative']
        
        # Multi-stage association
        
        # Stage 1: confirmed tracks with high confidence detections
        high_conf_dets = [i for i, d in enumerate(detections) if d[4] >= 0.7]
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            confirmed_tracks, detections, high_conf_dets, appearance_features
        )
        
        # Stage 2: remaining tracks with remaining detections
        remaining_dets = [i for i in range(len(detections)) if i not in unmatched_dets]
        low_conf_dets = [i for i in remaining_dets if i not in high_conf_dets]
        
        if len(unmatched_tracks) > 0 and len(low_conf_dets) > 0:
            matched_tracks_2, unmatched_tracks_2, unmatched_dets_2 = self._associate(
                [confirmed_tracks[i] if i < len(confirmed_tracks) else self.tracks[i] 
                 for i in unmatched_tracks], detections, low_conf_dets, appearance_features
            )
            # Convert back to global indices
            for track_idx, det_idx in matched_tracks_2:
                global_track_idx = unmatched_tracks[track_idx] if track_idx < len(unmatched_tracks) else track_idx
                matched_tracks.append((global_track_idx, det_idx))
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_2 if i < len(unmatched_tracks)]
            unmatched_dets.extend(unmatched_dets_2)
        
        # Stage 3: tentative tracks with remaining detections
        remaining_dets = [i for i in range(len(detections)) if i not in unmatched_dets]
        if len(tentative_tracks) > 0 and len(remaining_dets) > 0:
            # Find tentative track indices in main tracks list
            tentative_indices = [i for i, t in enumerate(self.tracks) if t.state == 'tentative']
            matched_tracks_3, unmatched_tentative, unmatched_dets_3 = self._associate(
                tentative_tracks, detections, remaining_dets, appearance_features
            )
            # Convert back to global indices
            for track_idx, det_idx in matched_tracks_3:
                global_track_idx = tentative_indices[track_idx] if track_idx < len(tentative_indices) else track_idx
                matched_tracks.append((global_track_idx, det_idx))
            unmatched_tracks.extend([tentative_indices[i] for i in unmatched_tentative if i < len(tentative_indices)])
            unmatched_dets.extend(unmatched_dets_3)
        else:
            tentative_indices = [i for i, t in enumerate(self.tracks) if t.state == 'tentative']
            unmatched_tracks.extend(tentative_indices)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            if track_idx < len(self.tracks) and det_idx < len(detections):
                track = self.tracks[track_idx]
                x, y, w, h, conf, cls = detections[det_idx][:6]
                appearance_feat = appearance_features[det_idx] if det_idx < len(appearance_features) else None
                track.update((x, y, w, h), conf, appearance_feat)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):
                self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        used_det_indices = set(det_idx for _, det_idx in matched_tracks)
        for det_idx in range(len(detections)):
            if det_idx not in used_det_indices:
                det = detections[det_idx]
                x, y, w, h, conf, cls = det[:6]
                appearance_feat = appearance_features[det_idx] if det_idx < len(appearance_features) else None
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
        """Associate tracks with detections using cost matrix"""
        if len(tracks) == 0 or len(detection_indices) == 0:
            return [], list(range(len(tracks))), detection_indices
        
        # Create cost matrix
        cost_matrix = []
        for i, track in enumerate(tracks):
            cost_row = []
            for j, det_idx in enumerate(detection_indices):
                if det_idx >= len(detections):
                    cost_row.append(1.0)
                    continue
                    
                det = detections[det_idx]
                x, y, w, h, conf, cls = det[:6]
                
                # IoU cost
                iou = self._iou(getattr(track, 'predicted_bbox', track.bbox), (x, y, w, h))
                iou_cost = 1.0 - iou
                
                # Appearance cost
                appearance_cost = 0.0
                if (appearance_features and det_idx < len(appearance_features) and 
                    appearance_features[det_idx] is not None and track.get_average_appearance() is not None):
                    appearance_sim = self._appearance_similarity(
                        track.get_average_appearance(), appearance_features[det_idx]
                    )
                    appearance_cost = 1.0 - appearance_sim
                
                # Combined cost
                total_cost = (1.0 - self.appearance_weight) * iou_cost + self.appearance_weight * appearance_cost
                
                # Set high cost for low IoU
                if iou < self.iou_thresh:
                    total_cost = 1.0
                
                cost_row.append(total_cost)
            cost_matrix.append(cost_row)
        
        # Simple assignment - try Hungarian-like approach
        matched_tracks = []
        unmatched_tracks = []
        unmatched_dets = []
        
        used_tracks = set()
        used_dets = set()
        
        # Greedy assignment based on lowest cost
        for _ in range(min(len(tracks), len(detection_indices))):
            min_cost = float('inf')
            best_match = None
            
            for i in range(len(tracks)):
                if i in used_tracks:
                    continue
                for j in range(len(detection_indices)):
                    if j in used_dets:
                        continue
                    if cost_matrix[i][j] < min_cost and cost_matrix[i][j] < 1.0:
                        min_cost = cost_matrix[i][j]
                        best_match = (i, j)
            
            if best_match:
                i, j = best_match
                matched_tracks.append((i, detection_indices[j]))
                used_tracks.add(i)
                used_dets.add(j)
            else:
                break
        
        # Find unmatched tracks and detections
        for i in range(len(tracks)):
            if i not in used_tracks:
                unmatched_tracks.append(i)
        
        for j in range(len(detection_indices)):
            if j not in used_dets:
                unmatched_dets.append(detection_indices[j])
        
        return matched_tracks, unmatched_tracks, unmatched_dets
    
    def _extract_appearance_features(self, detections, frame):
        """Extract simple appearance features from detections"""
        features = []
        
        try:
            import cv2
            for det in detections:
                x, y, w, h = det[:4]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    features.append([0.0] * 12)  # Default feature
                    continue
                
                roi = frame[y1:y2, x1:x2]
                
                # Simple color features - mean values
                if len(roi.shape) == 3:
                    mean_b = float(roi[:, :, 0].mean()) if roi.size > 0 else 0.0
                    mean_g = float(roi[:, :, 1].mean()) if roi.size > 0 else 0.0
                    mean_r = float(roi[:, :, 2].mean()) if roi.size > 0 else 0.0
                    
                    # Standard deviation for texture
                    std_b = float(roi[:, :, 0].std()) if roi.size > 0 else 0.0
                    std_g = float(roi[:, :, 1].std()) if roi.size > 0 else 0.0
                    std_r = float(roi[:, :, 2].std()) if roi.size > 0 else 0.0
                    
                    # Aspect ratio and size features
                    aspect_ratio = w / h if h > 0 else 1.0
                    size_feature = (w * h) / (frame.shape[0] * frame.shape[1])
                    
                    # Position features (normalized)
                    center_x = (x + w/2) / frame.shape[1]
                    center_y = (y + h/2) / frame.shape[0]
                    
                    feature = [mean_b, mean_g, mean_r, std_b, std_g, std_r, 
                              aspect_ratio, size_feature, center_x, center_y, w/frame.shape[1], h/frame.shape[0]]
                else:
                    feature = [0.0] * 12
                
                features.append(feature)
        except:
            # Fallback to dummy features if CV2 not available
            features = [[0.0] * 12 for _ in detections]
        
        return features
    
    def _appearance_similarity(self, feat1, feat2):
        """Calculate appearance similarity using simple correlation"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            # Simple cosine similarity
            dot_product = sum(a * b for a, b in zip(feat1, feat2))
            norm1 = sum(a * a for a in feat1) ** 0.5
            norm2 = sum(b * b for b in feat2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return max(0.0, min(1.0, dot_product / (norm1 * norm2)))
        except:
            return 0.0
    
    def _iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU)"""
        try:
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
        except:
            return 0.0

# Backward compatibility
BYTETracker = SimpleBYTETracker