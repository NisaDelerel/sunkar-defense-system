import pygame
import time

pygame.init()
pygame.joystick.init()

joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("Joystick bulunamadı!")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick bulundu: {joystick.get_name()}")
    print(f"Eksen sayısı: {joystick.get_numaxes()}")
    print(f"Buton sayısı: {joystick.get_numbuttons()}")
    print(f"Hat sayısı: {joystick.get_numhats()}")

    try:
        while True:
            pygame.event.pump()

            axes = [round(joystick.get_axis(i), 2) for i in range(joystick.get_numaxes())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

            print(f"Axes: {axes} | Buttons: {buttons} | Hats: {hats}", end='\r', flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nÇıkış yapıldı.")

    finally:
        pygame.quit()
