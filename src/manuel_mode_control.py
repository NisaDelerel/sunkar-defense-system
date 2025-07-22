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
