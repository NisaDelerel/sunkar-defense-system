import time
from camera_manager import CameraManager
from laser_control import LaserControl

class ManualModeControl:
    def __init__(self, camera_manager, laser_control, joystick):
        self.camera_manager = camera_manager
        self.laser_control = laser_control
        self.joystick = joystick
        self.is_manual_mode = False
        self.target_centered = False

    def switch_to_manual(self):
        self.is_manual_mode = True
        print("Manuel Mod Başlatıldı.")

    def joystick_input(self):
        # Joystick'in X, Y konum verilerini al
        x, y = self.joystick.get_position()
        self.camera_manager.move_camera(x, y)

    def center_target(self):
        # Kamerayı hedefe odakla
        if self.camera_manager.is_target_centered():
            self.target_centered = True
            print("Hedef Ortalanmış.")
        else:
            self.target_centered = False

    def fire_laser(self):
        if self.target_centered:
            self.laser_module.fire_laser()
            return "Ateş Başarılı!"
        else:
            return "Hedef Ortalanmadı, Ateş Edilemez!"

    def is_balloon_centered(self, selected_bbox, frame_shape, threshold=20):
        """
        Checks if the center of the selected balloon's bbox is close to the center of the frame.
        selected_bbox: (x1, y1, x2, y2)
        frame_shape: (height, width, channels)
        """
        if selected_bbox is None:
            return False
        x1, y1, x2, y2 = selected_bbox
        balloon_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
        dx = abs(balloon_center[0] - frame_center[0])
        dy = abs(balloon_center[1] - frame_center[1])
        return dx < threshold and dy < threshold
