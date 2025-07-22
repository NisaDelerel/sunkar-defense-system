import serial
import time

class SerialComm:
    def __init__(self, port="/dev/serial0", baudrate=9600, timeout=1):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Arduino reset sonrası bekleme süresi
            print(f"[SerialComm] Port {port} açıldı, baudrate: {baudrate}")
        except serial.SerialException as e:
            print(f"[SerialComm] Seri port açılamadı: {e}")
            self.ser = None

    def send_command(self, cmd, data):
        if self.ser and self.ser.is_open:
            packet = bytearray([0xAA, cmd, data, 0x55])
            self.ser.write(packet)
            print(f"[SerialComm] Gönderildi: CMD={cmd}, DATA={data}")
        else:
            print("[SerialComm] Port açık değil, gönderim yapılamadı.")

    def read_response(self):
        if self.ser and self.ser.in_waiting:
            response = self.ser.readline().decode(errors='ignore').strip()
            print(f"[SerialComm] Gelen cevap: {response}")
            return response
        return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialComm] Port kapatıldı.")

# Modül test örneği:
if __name__ == "__main__":
    comm = SerialComm()
    comm.send_command(0x01, 90)  # Örnek: servo 90 dereceye
    time.sleep(1)
    comm.send_command(0x03, 1)   # Örnek: lazer aç
    time.sleep(1)
    comm.send_command(0x03, 0)   # Örnek: lazer kapat
    comm.close()
