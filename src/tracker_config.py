# tracker_config.py
"""
Configuration file for BYTETracker parameters.
Adjust these values to fine-tune tracking performance for your specific use case.
"""

class TrackerConfig:
    # Detection parameters
    YOLO_CONF_THRESH = 0.6          # YOLO confidence threshold
    MIN_DETECTION_SIZE = 20         # Minimum width/height for valid detections
    
    # Tracker parameters
    IOU_THRESH = 0.6                # IoU threshold for track-detection matching
    MAX_MISSED = 30                 # Maximum frames a track can be missed before deletion
    TRACKER_CONF_THRESH = 0.5       # Minimum confidence for detections to be tracked
    
    # Track lifecycle parameters
    N_INIT = 3                      # Number of consecutive detections to confirm track
    MAX_AGE_TENTATIVE = 1           # Max age for tentative tracks
    MAX_AGE_CONFIRMED = 30          # Max age for confirmed tracks
    
    # Appearance feature parameters
    APPEARANCE_WEIGHT = 0.3         # Weight for appearance similarity (0.0 = IoU only, 1.0 = appearance only)
    MAX_APPEARANCE_FEATURES = 10    # Number of appearance features to store per track
    
    # Kalman filter parameters
    PROCESS_NOISE = 0.1             # Process noise for Kalman filter
    MEASUREMENT_NOISE = 1.0         # Measurement noise for Kalman filter
    
    # Visualization parameters
    HIGH_CONF_THRESH = 0.8          # Threshold for high confidence visualization
    MED_CONF_THRESH = 0.6           # Threshold for medium confidence visualization
    
    # Color scheme (BGR format)
    COLOR_HIGH_CONF = (0, 255, 0)       # Green for high confidence
    COLOR_MED_CONF = (0, 255, 255)      # Yellow for medium confidence  
    COLOR_LOW_CONF = (0, 165, 255)      # Orange for low confidence

# Additional configurations for specific scenarios
class BalloonTrackingConfig(TrackerConfig):
    """Optimized configuration for balloon tracking"""
    # Balloons can move quickly and unpredictably
    IOU_THRESH = 0.5                # Lower IoU for faster moving objects
    MAX_MISSED = 20                 # Shorter retention for dynamic scenes
    APPEARANCE_WEIGHT = 0.4         # Higher appearance weight for balloon colors
    PROCESS_NOISE = 0.2             # Higher process noise for unpredictable movement

# Usage example:
# config = BalloonTrackingConfig()
# tracker = BYTETracker(
#     iou_thresh=config.IOU_THRESH,
#     max_missed=config.MAX_MISSED,
#     conf_thresh=config.TRACKER_CONF_THRESH
# )