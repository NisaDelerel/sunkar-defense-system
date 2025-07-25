import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image
import cv2
import threading
import time

from manuel_mode_control import ManualModeControl
from laser_control import LaserControl
from serial_comm import SerialComm
from joystick_controller import JoystickController

class SunkarGUI(ctk.CTk):
    def __init__(self, camera_manager):
        super().__init__()
        self.title("SUNKAR Defense Interface")
        self.geometry("1280x680")
        self.resizable(False, False)
        self.camera_manager = camera_manager
        self.manual_control = None
        self.serial_comm = SerialComm(port="COM3")  # Arduino’nun bağlı olduğu doğru portu yaz
        self.laser_control = LaserControl(self.serial_comm)
        self.joystick = JoystickController(port="COM3", mode="manual")
        self.selected_track_id = None
        self.last_joystick_button_state = False
        self.bind("<Button-1>", self.on_video_click)  # Mouse click event
        self.auto_fired_ids = set()  # Track which balloons have been fired at in auto mode
        self.auto_mode_active = False
        self.crosshair_timer = 0  # Number of frames to keep crosshair visible
        self.crosshair_bbox = None  # The bbox to keep crosshair on
        self.crosshair_track_id = None  # Track ID to keep crosshair on in auto mode

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#1C222D")

        # Ana container (pencerenin ortasında olacak)
        self.main_container = ctk.CTkFrame(self, width=1140, height=600, fg_color="#131820", corner_radius=24)
        self.main_container.place(relx=0.5, rely=0.5, anchor="center")

        # 3 sütunlu grid
        self.main_container.grid_columnconfigure((0, 1, 2), weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Sol taraf: Kamera + alt kısımlar için Frame (2 sütun kaplayacak)
        self.camera_frame = ctk.CTkFrame(self.main_container, fg_color="#131820", corner_radius=24)
        self.camera_frame.grid(row=0, column=0, columnspan=2, padx=(20, 10), pady=20, sticky="nsew")

        # Kamera görüntüsü
        self.video_label = ctk.CTkLabel(self.camera_frame, text="", width=700, height=400, corner_radius=12)
        self.video_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10))

        # Sistem durumu
        self.status_box = ctk.CTkLabel(self.camera_frame, text="Sistem Başlatıldı.", width=340, height=80,
                                       fg_color="#242E3A", corner_radius=12, font=("Inter", 20),
                                       anchor="center", justify="center")
        self.status_box.grid(row=1, column=0, padx=(20, 10), pady=(50, 30), sticky="nsew")

        # Zoom butonları çerçevesi
        self.zoom_frame = ctk.CTkFrame(self.camera_frame, width=340, height=80, fg_color="#242E3A", corner_radius=12)
        self.zoom_frame.grid(row=1, column=1, padx=(10, 20), pady=(50, 30), sticky="nsew")

        # Zoom butonlarını grid ile ortala
        self.zoom_frame.grid_columnconfigure((0, 1), weight=1)
        self.zoom_frame.grid_rowconfigure(0, weight=1)
        
        self.zoom_in = ctk.CTkButton(self.zoom_frame, text="Yaklaştır", font=("Inter", 20), height=50)
        self.zoom_in.grid(row=0, column=0, padx=10, pady=15, sticky="ew")

        self.zoom_out = ctk.CTkButton(self.zoom_frame, text="Uzaklaştır", font=("Inter", 20), height=50)
        self.zoom_out.grid(row=0, column=1, padx=10, pady=15, sticky="ew")

        # Sağ taraf: Kontrol paneli için Frame (1 sütun)
        self.panel = ctk.CTkFrame(self.main_container, width=320, height=540, fg_color="#131820", corner_radius=24)
        self.panel.grid(row=0, column=2, padx=(5, 20), pady=20, sticky="n")

        # Kontrol Modu başlığı
        self.control_label = ctk.CTkLabel(self.panel, text="Kontrol Modu", font=("Inter", 28, "bold"), text_color="#FFFFFF")
        self.control_label.place(x=20, y=20)

        # Manuel / Auto switch ve label'ları
        self.manual_label = ctk.CTkLabel(self.panel, text="Manuel", font=("Inter", 28), text_color="#8892A6")
        self.manual_label.place(x=20, y=70)

        self.mode_switch = ctk.CTkSwitch(self.panel, text="", width=70)
        self.mode_switch.place(x=130, y=75)

        self.auto_label = ctk.CTkLabel(self.panel, text="Auto", font=("Inter", 28), text_color="#8892A6")
        self.auto_label.place(x=200, y=70)

        # Başlat butonu
        self.start_button = ctk.CTkButton(self.panel, text="Başlat", width=140, height=45,
                                          fg_color="#242E3A", font=("Inter", 28), corner_radius=12, command=self.start_system)
        self.start_button.place(x=20, y=130)

        # Başlangıç Konumuna Getir butonu (iki satır yazı, aynı genişlikte olacak)
        self.reset_button = ctk.CTkButton(self.panel, text="Başlangıç\nKonumuna Getir", width=140, height=45,
                                          fg_color="#242E3A", font=("Inter", 20), corner_radius=12, command=self.reset_position)
        self.reset_button.place(x=170, y=130)

        # Ateş Et butonu
        self.fire_button = ctk.CTkButton(self.panel, text="Ateş Et", width=290, height=50,
                                         fg_color="#EF4C4C", text_color="#FFFFFF",
                                         font=("Inter", 28, "bold"), corner_radius=12, command=self.fire_action)
        self.fire_button.place(x=20, y=200)

        # Ayırıcı çizgi
        self.separator1 = ctk.CTkFrame(self.panel, height=3, width=290, fg_color="#1C222D")
        self.separator1.place(x=20, y=260)

        # Yasak Alan Kontrolleri butonu
        self.restricted_button = ctk.CTkButton(self.panel, text="Yasak Alan Kontrolleri", width=290, height=45,
                                               fg_color="#242E3A", font=("Inter", 20), text_color="#FFFFFF",
                                               corner_radius=12,  command=self.restricted_area)
        self.restricted_button.place(x=20, y=280)

        # Angajman Kabul Et butonu
        self.engage_button = ctk.CTkButton(self.panel, text="Angajman Kabul Et", width=290, height=50,
                                           fg_color="#EF4C4C", font=("Inter", 28), text_color="#FFFFFF",
                                           corner_radius=12, command=self.engage_action)
        self.engage_button.place(x=20, y=340)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_camera()

    def start_camera(self):
        self.camera_manager.start()
        self.update_loop()

    def update_loop(self):
        frame, tracks = self.camera_manager.get_frame()
        if frame is not None:
            auto_mode = self.mode_switch.get() == 1
            selected_bbox = None

            # --- Always draw bounding boxes for all detections ---
            for det in tracks:
                x1, y1, x2, y2 = det['bbox']
                track_id = det['track_id']
                color = (0, 255, 0)
                # Highlight the selected target in red
                if track_id == self.selected_track_id:
                    color = (0, 0, 255)
                    selected_bbox = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {det['label']}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Autonomous targeting logic with debug prints ---
            if auto_mode:
                red_balloons = [det for det in tracks if det['label'].lower() == "red"]
                print(f"[AUTO] Detected red balloons: {[{'id': b['track_id'], 'bbox': b['bbox']} for b in red_balloons]}")
                if red_balloons:
                    candidates = [b for b in red_balloons if b['track_id'] not in self.auto_fired_ids]
                    if candidates:
                        target = min(candidates, key=lambda d: d['track_id'])
                        self.selected_track_id = target['track_id']
                        selected_bbox = target['bbox']
                        print(f"[AUTO] Targeting balloon: track_id={target['track_id']}, bbox={target['bbox']}")
                        # Fire laser only once per target
                        if target['track_id'] not in self.auto_fired_ids:
                            self.auto_fired_ids.add(target['track_id'])
                            threading.Thread(target=self.auto_fire_laser, args=(0.5,), daemon=True).start()
                            # Keep crosshair on this track_id
                            self.crosshair_track_id = target['track_id']
                        self.status_box.configure(text=f"Otonom: Hedeflenen balon ID {target['track_id']}")
                    else:
                        print("[AUTO] All red balloons have been targeted already.")
                        self.status_box.configure(text="Otonom: Kırmızı balon kalmadı.")
                        self.selected_track_id = None
                else:
                    print("[AUTO] No red balloons detected in this frame.")
                    self.status_box.configure(text="Otonom: Kırmızı balon yok.")
                    self.selected_track_id = None
            else:
                # Joystick button for cycling selection
                button_index = 2  # Example: X button index
                button_pressed = self.joystick.get_button_pressed(button_index)
                if button_pressed and not self.last_joystick_button_state:
                    self.cycle_selected_balloon(tracks)
                self.last_joystick_button_state = button_pressed

                # Check B button (index 1) for firing
                b_button_index = 1
                b_button_pressed = self.joystick.get_button_pressed(b_button_index)
                if b_button_pressed and not getattr(self, 'last_b_button_state', False):
                    self.fire_action()
                self.last_b_button_state = b_button_pressed

            # --- Draw crosshair for selected target (auto or manual) ---
            crosshair_drawn = False
            if selected_bbox is not None:
                self.draw_crosshair(frame, ((selected_bbox[0] + selected_bbox[2]) // 2, (selected_bbox[1] + selected_bbox[3]) // 2))
                crosshair_drawn = True
            # In auto mode, keep crosshair on last targeted object as long as it is present
            if auto_mode and not crosshair_drawn and self.crosshair_track_id is not None:
                # Find the bbox for the last targeted track_id
                for det in tracks:
                    if det['track_id'] == self.crosshair_track_id:
                        bbox = det['bbox']
                        cx = (bbox[0] + bbox[2]) // 2
                        cy = (bbox[1] + bbox[3]) // 2
                        self.draw_crosshair(frame, (cx, cy))
                        crosshair_drawn = True
                        break
                # If the object is no longer present, remove the crosshair
                if not crosshair_drawn:
                    self.crosshair_track_id = None

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img)
            imgtk = CTkImage(light_image=pil_image, size=(700, 400))
            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk
        self.after(30, self.update_loop)

    def auto_fire_laser(self, duration=0.5):
        self.laser_control.turn_on()
        time.sleep(duration)
        self.laser_control.turn_off()

    def start_joystick_loop(self):
        def joystick_loop():
            while self.manual_control.is_manual_mode:
                self.manual_control.joystick_input()
                time.sleep(0.1)  # 10 Hz joystick polling rate
        threading.Thread(target=joystick_loop, daemon=True).start()

    def start_system(self):
        mode = self.mode_switch.get()
        if mode == 0:
            self.status_box.configure(text="Manuel Mod Başlatıldı.")
            print("Manuel mod başlatılıyor...")
            self.auto_fired_ids.clear()
            self.auto_mode_active = False
            # ManualModeControl başlat
            self.manual_control = ManualModeControl(self.camera_manager, self.laser_control, self.joystick)
            self.manual_control.switch_to_manual()
            self.start_joystick_loop()
        else:
            self.status_box.configure(text="Otonom Mod Başlatıldı.")
            print("Otonom mod başlatılıyor...")
            self.auto_fired_ids.clear()
            self.auto_mode_active = True

    def reset_position(self):
        self.camera_manager.reset_position()
        self.status_box.configure(text="Başlangıç konumuna getirildi.")
        print("Reset butonu çalıştı.")

    def draw_crosshair(self, frame, center, color=(0,0,255), size=10, thickness=2):
        x, y = center
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(frame, (x, y), 3, color, -1)

    def on_video_click(self, event):
        frame, tracks = self.camera_manager.get_frame()
        if frame is None:
            return

        display_w, display_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        frame_h, frame_w = frame.shape[:2]

        x_img = int(event.x * frame_w / display_w)
        y_img = int(event.y * frame_h / display_h)

        for det in tracks:
            x1, y1, x2, y2 = det['bbox']
            if x1 <= x_img <= x2 and y1 <= y_img <= y2:
                self.selected_track_id = det['track_id']
                break

    def cycle_selected_balloon(self, tracks):
        track_ids = [det['track_id'] for det in tracks]
        if not track_ids:
            self.selected_track_id = None
            return
        if self.selected_track_id not in track_ids:
            self.selected_track_id = track_ids[0]
        else:
            idx = track_ids.index(self.selected_track_id)
            self.selected_track_id = track_ids[(idx + 1) % len(track_ids)]

    def fire_action(self):
        frame, tracks = self.camera_manager.get_frame()
        selected_bbox = None
        for det in tracks:
            if det['track_id'] == self.selected_track_id:
                selected_bbox = det['bbox']
                break
        if selected_bbox and self.manual_control.is_balloon_centered(selected_bbox, frame.shape):
            self.manual_control.fire_laser()
            self.status_box.configure(text="ATEŞ EDİLDİ!")
        else:
            self.status_box.configure(text="Ateş edilemez, hedef ortalanmadı.")

    def restricted_area(self):
        self.status_box.configure(text="Yasak Alan Kontrolleri aktif.")
        print("Yasak Alan Kontrolleri butonu çalıştı.")

    def engage_action(self):
        self.status_box.configure(text="Angajman kabul edildi.")
        print("Angajman Kabul Et butonu çalıştı.")

    def on_close(self):
        self.camera_manager.stop()
        self.destroy() 
