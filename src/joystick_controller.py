import pygame
import time
from laser_control import LaserControl
from serial_comm import SerialComm
from otonom_mode_control import OtonomModeControl  # Otonom algoritmasını import et

class JoystickController:
    def __init__(self, port="COM3", mode="manual"):
        self.serial = SerialComm(port=port)
        self.mode = mode  # Mod: "manual" veya "auto"
        self.lazer_control = LaserControl(self.serial)  # Lazer kontrolü
        self.otonom_mode = OtonomModeControl(self.serial)  # Otonom algoritması

        pygame.init()
        pygame.joystick.init()
        self.joystick = None

        if pygame.joystick.get_count() == 0:
            print("[JoystickController] Joystick bulunamadı!")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[JoystickController] Joystick başlatıldı: {self.joystick.get_name()}")

    def set_mode(self, mode):
        self.mode = mode

    def manual_mode_control(self):
        """Manuel modda joystick hareketleriyle lazer ve motor kontrolü."""
        if self.joystick:
            x_axis = self.joystick.get_axis(0)  # X ekseni (lazer gücü)
            y_axis = self.joystick.get_axis(1)  # Y ekseni (lazer açısı)

            lazer_power = int((x_axis + 1) * 127.5)  # -1 to 1 arası değerleri 0-255'e dönüştür
            lazer_angle = int((y_axis + 1) * 90)    # -1 to 1 arası değerleri 0-180'ye dönüştür

            # Lazerin gücünü ve açısını ayarla
            self.lazer_control.set_laser(lazer_power)
            # Burada lazer açısını kontrol etmek isterseniz, uygun bir servo komutu da ekleyebilirsiniz
            print(f"[JoystickController] Lazer gücü: {lazer_power}, Lazer açısı: {lazer_angle}")

    def get_position(self):
        if self.joystick:
            pygame.event.pump()
            x_axis = self.joystick.get_axis(0)
            y_axis = self.joystick.get_axis(1)
            return (x_axis, y_axis)
        else:
            return (0, 0)

    def get_button_pressed(self, button_index):
        """
        Returns True if the specified joystick button is pressed.
        """
        if self.joystick:
            pygame.event.pump()
            return self.joystick.get_button(button_index)
        return False

    def run(self):
        """Joystick kontrolünü başlat."""
        if not self.joystick and self.mode == "manual":
            print("[JoystickController] Joystick yok, çıkılıyor.")
            return

        try:
            while True:
                pygame.event.pump()

                if self.mode == "manual":
                    # Manuel modda joystick hareketleriyle motor ve lazer kontrolü
                    self.manual_mode_control()
                elif self.mode == "auto":
                    # Otonom modda joystick devre dışı kalacak
                    print("[JoystickController] Otonom modda çalışıyor.")
                    self.otonom_mode.start()  # Otonom mod algoritmasını çalıştır

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[JoystickController] Kontrol durduruldu.")
        finally:
            self.serial.close()
            pygame.quit()