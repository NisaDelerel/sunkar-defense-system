from camera_manager import CameraManager
from gui import SunkarGUI

if __name__ == "__main__":
    cam = CameraManager() 
    app = SunkarGUI(cam)
    app.mainloop()