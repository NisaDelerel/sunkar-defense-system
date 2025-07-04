import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image
import cv2

class SunkarGUI(ctk.CTk):
    def __init__(self, camera_manager):
        super().__init__()
        self.title("SUNKAR Defense Interface")
        self.geometry("1280x680")
        self.resizable(False, False)
        self.camera_manager = camera_manager

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

        self.zoom_in = ctk.CTkButton(self.zoom_frame, text="Yaklaştır", font=("Inter", 20))
        self.zoom_in.grid(row=0, column=0, padx=10, pady=15, sticky="ew")

        self.zoom_out = ctk.CTkButton(self.zoom_frame, text="Uzaklaştır", font=("Inter", 20))
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
                                          fg_color="#242E3A", font=("Inter", 28), corner_radius=12)
        self.start_button.place(x=20, y=130)

        # Başlangıç Konumuna Getir butonu (iki satır yazı, aynı genişlikte olacak)
        self.reset_button = ctk.CTkButton(self.panel, text="Başlangıç\nKonumuna Getir", width=140, height=45,
                                          fg_color="#242E3A", font=("Inter", 20), corner_radius=12)
        self.reset_button.place(x=170, y=130)

        # Ateş Et butonu
        self.fire_button = ctk.CTkButton(self.panel, text="Ateş Et", width=290, height=50,
                                         fg_color="#EF4C4C", text_color="#FFFFFF",
                                         font=("Inter", 28, "bold"), corner_radius=12)
        self.fire_button.place(x=20, y=200)

        # Ayırıcı çizgi
        self.separator1 = ctk.CTkFrame(self.panel, height=3, width=290, fg_color="#1C222D")
        self.separator1.place(x=20, y=260)

        # Yasak Alan Kontrolleri butonu
        self.restricted_button = ctk.CTkButton(self.panel, text="Yasak Alan Kontrolleri", width=290, height=45,
                                               fg_color="#242E3A", font=("Inter", 20), text_color="#FFFFFF",
                                               corner_radius=12)
        self.restricted_button.place(x=20, y=280)

        # Angajman Kabul Et butonu
        self.engage_button = ctk.CTkButton(self.panel, text="Angajman Kabul Et", width=290, height=50,
                                           fg_color="#EF4C4C", font=("Inter", 28), text_color="#FFFFFF",
                                           corner_radius=12)
        self.engage_button.place(x=20, y=340)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_camera()

    def start_camera(self):
        self.camera_manager.start()
        self.update_loop()

    def update_loop(self):
        frame, tracks = self.camera_manager.get_frame()

        if frame is not None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img)
            imgtk = CTkImage(light_image=pil_image, size=(700, 400))
            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk
        self.after(30, self.update_loop)

    def on_close(self):
        self.camera_manager.stop()
        self.destroy()
