class LaserControl:
    def __init__(self, serial_comm):
        self.serial = serial_comm

    def adjust_laser_power(self, power):
        """PWM ile lazerin gücünü ayarlama (0-255 arasında)."""
        if power < 0 or power > 255:
            print("[LazerControl] Geçersiz güç değeri. 0-255 arası olmalı.")
            return
        self.serial.send_command(0x04, power)  # 0x04 lazer kontrolü için komut
        print(f"[LazerControl] Lazer gücü: {power}")
        
    def turn_on(self):
        """Lazer açma."""
        self.adjust_laser_power(255)  # Lazer tam güçle açılır
        
    def turn_off(self):
        """Lazer kapama."""
        self.adjust_laser_power(0)  # Lazer kapalı