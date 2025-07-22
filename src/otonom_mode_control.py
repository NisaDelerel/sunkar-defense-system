# otonom_mode_control.py

class OtonomModeControl:
    def __init__(self, serial_comm):
        self.serial = serial_comm

    def start(self):
        print("[OtonomModeControl] Otonom mod başlatıldı.")
        # Buraya gerçek otonom mod algoritması eklenecek
