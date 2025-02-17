import sys
import threading
import time
import platform
import logging
import keyboard
import json
import pyautogui
import cv2
import numpy as np
import torch
import mss
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor


# ------------------------- Compatibility Check -------------------------
def check_compatibility():
    if platform.system() != "Windows" or platform.release() not in ["10", "11"]:
        print("Incompatible OS. This cheat only supports Windows 10/11.")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA not available. Falling back to CPU.")
    
    cpu_info = platform.processor()
    if "Intel" in cpu_info or "AMD" in cpu_info:
        print(f"Using CPU: {cpu_info}")
    else:
        print("Warning: Unsupported CPU model detected. Performance may be impacted.")
    
    return cuda_available

# ------------------------- Logging Setup -------------------------
logging.basicConfig(filename='cheat_log.txt',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.info("Dark Market AI Loaded.")

# ------------------------- Cheat Settings -------------------------
cheat_settings = {
    "aimbot_enabled": True,
    "softaim_active": True,
    "aimbot_smoothing": 0.2,
    "conf_threshold": 0.995,
    "anti_recoil_enabled": True,
    "anti_recoil_intensity": 6,
    "crouch_spam_enabled": True,
    "crouch_frequency": 0.2,
    "draw_fov_circle": True,
    "fov_circle_radius": 200,
    "draw_boxes_enabled": True,
    "draw_skeleton_enabled": True,
    "skeleton_thickness": 2,
    "weapon_slots": [1, 2, 3, 4, 5, 6],
    "current_slot_index": 0,
    "toggle_hotkey": "Insert",
    "exit_hotkey": "F9",
    "softaim_key": "F7",  # Toggle softaim
    "learning_rate": 0.05,  # AI learning rate
}

# ------------------------- GPU Setup -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Device: {device} - {'GPU' if device == 'cuda' else 'CPU'}")

model = torch.hub.load('custom_yolov10_repo', 'custom', path='yolov10_custom.pt', force_reload=True)
model.to(device)
model.eval()

# ------------------------- Screen Capture -------------------------
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
screen_width = monitor['width']
screen_height = monitor['height']

# ------------------------- Configuration Save/Load -------------------------
def save_settings():
    with open("cheat_settings.json", "w") as f:
        json.dump(cheat_settings, f, indent=4)
    logging.info("Settings saved.")

def load_settings():
    global cheat_settings
    try:
        with open("cheat_settings.json", "r") as f:
            cheat_settings = json.load(f)
        logging.info("Settings loaded.")
    except FileNotFoundError:
        logging.warning("Settings file not found. Using default settings.")

# ------------------------- AI Learning -------------------------
class AILearning:
    def __init__(self):
        self.learning_rate = cheat_settings["learning_rate"]
        self.previous_positions = []
        self.target_predictions = []

    def update_target(self, target_position):
        self.previous_positions.append(target_position)
        if len(self.previous_positions) > 100:
            self.previous_positions.pop(0)

    def predict_target(self):
        if len(self.previous_positions) < 2:
            return None

        last_position = self.previous_positions[-1]
        second_last_position = self.previous_positions[-2]
        direction = np.subtract(last_position, second_last_position)
        predicted_position = np.add(last_position, direction)
        self.target_predictions.append(predicted_position)
        return predicted_position

# ------------------------- Cheat Core Functions -------------------------
def capture_screen(mon):
    with mss.mss() as sct:
        frame = np.array(sct.grab(mon))
    return frame

def preprocess(frame):
    if not torch.cuda.is_available():
        frame = cv2.resize(frame, (960, 540))  # Lower resolution for CPU
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return rgb_frame

def detect_targets(frame):
    img_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    results = model(img_tensor)
    detections = results.xyxy[0].cpu().detach().numpy()
    return detections

def select_best_target(detections, threshold):
    valid_targets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf >= threshold:
            valid_targets.append(det)
    if not valid_targets:
        return None
    center_x, center_y = screen_width / 2, screen_height / 2
    def score(det):
        x1, y1, x2, y2, conf, _ = det
        target_x = (x1 + x2) / 2
        target_y = (y1 + y2) / 2
        distance = np.hypot(center_x - target_x, center_y - target_y)
        return distance - (conf * 100)
    return min(valid_targets, key=score)

def smooth_aim(current, target, factor):
    return current + (target - current) * factor

def aim_at_target(det, factor):
    x1, y1, x2, y2, _, _ = det
    target_x = int((x1 + x2) / 2)
    target_y = int((y1 + y2) / 2)
    current_x, current_y = pyautogui.position()
    new_x = smooth_aim(current_x, target_x, factor)
    new_y = smooth_aim(current_y, target_y, factor)
    pyautogui.moveTo(new_x, new_y, duration=0.005)

def trigger_shot():
    pyautogui.click(button='left')

def apply_anti_recoil(intensity):
    current_x, current_y = pyautogui.position()
    pyautogui.moveTo(current_x, current_y - intensity, duration=0.01)

def crouch_spam(frequency):
    pyautogui.keyDown('ctrl')
    time.sleep(0.05)
    pyautogui.keyUp('ctrl')
    time.sleep(frequency)

def cycle_weapon_slot():
    cheat_settings['current_slot_index'] = (cheat_settings['current_slot_index'] + 1) % len(cheat_settings["weapon_slots"])
    slot = cheat_settings["weapon_slots"][cheat_settings['current_slot_index']]
    pyautogui.press(str(slot))
    logging.info(f"Switched to weapon slot {slot}")

# ------------------------- Cheat Loop -------------------------
def cheat_loop():
    ai_learning = AILearning()
    while True:
        if cheat_settings["aimbot_enabled"] and cheat_settings["softaim_active"]:
            frame = capture_screen(monitor)
            processed = preprocess(frame)
            detections = detect_targets(processed)
            target = select_best_target(detections, cheat_settings["conf_threshold"])
            if target is not None:
                ai_learning.update_target((target[0], target[1]))
                predicted_target = ai_learning.predict_target()

                if predicted_target is not None:
                    aim_at_target(predicted_target, cheat_settings["aimbot_smoothing"])
                else:
                    aim_at_target(target, cheat_settings["aimbot_smoothing"])
                
                trigger_shot()
                if cheat_settings["anti_recoil_enabled"]:
                    apply_anti_recoil(cheat_settings["anti_recoil_intensity"])
                if cheat_settings["crouch_spam_enabled"]:
                    crouch_spam(cheat_settings["crouch_frequency"])
            else:
                cycle_weapon_slot()
        time.sleep(0.005)

# ------------------------- Hotkey Listener -------------------------
def hotkey_listener():
    while True:
        if keyboard.is_pressed(cheat_settings["toggle_hotkey"]):
            cheat_settings["aimbot_enabled"] = not cheat_settings["aimbot_enabled"]
            logging.info(f"Aimbot toggled to {cheat_settings['aimbot_enabled']}")
            time.sleep(0.5)
        if keyboard.is_pressed(cheat_settings["softaim_key"]):
            cheat_settings["softaim_active"] = not cheat_settings["softaim_active"]
            logging.info(f"Softaim toggled to {cheat_settings['softaim_active']}")
            time.sleep(0.5)
        if keyboard.is_pressed(cheat_settings["exit_hotkey"]):
            logging.info("Exit key pressed. Shutting down cheat.")
            sys.exit(0)
        time.sleep(0.01)

# ------------------------- UI Setup -------------------------
class CheatUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dark Market AI")
        self.setGeometry(200, 200, 1000, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2a2a2a;
                border-radius: 15px;
                color: white;
                font-family: "Arial", sans-serif;
            }
            QPushButton {
                background-color: #444444;
                border-radius: 8px;
                border: 2px solid #666666;
                color: white;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
                border-color: #00ffea;
                transition: background-color 0.3s, border-color 0.3s;
            }
            QSlider::handle:horizontal {
                background: #00ffea;
                border-radius: 10px;
            }
            QLabel {
                font-size: 18px;
                color: white;
                padding: 5px;
            }
            QTabWidget {
                background-color: #333333;
                font-size: 16px;
            }
            QTabBar::tab {
                background-color: #444444;
                padding: 12px;
                border-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #00ffea;
                border-color: #00ffea;
            }
        """)

        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)
        
        # Creating tabs for different cheat functionalities
        self.createTabs()
        
    def createTabs(self):
        # AimBot Tab
        aimbot_tab = QtWidgets.QWidget()
        aimbot_layout = QtWidgets.QVBoxLayout()
        
        # Aimbot Enable Button
        self.aimbot_enable_button = QtWidgets.QPushButton("Enable Aimbot", self)
        aimbot_layout.addWidget(self.aimbot_enable_button)
        
        # Distance Slider
        self.aimbot_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.aimbot_distance_slider.setRange(0, 500)
        aimbot_layout.addWidget(self.aimbot_distance_slider)
        
        aimbot_tab.setLayout(aimbot_layout)
        self.tabs.addTab(aimbot_tab, "AimBot")
        
        # Visuals Tab
        visuals_tab = QtWidgets.QWidget()
        visuals_layout = QtWidgets.QVBoxLayout()
        
        # ESP Enable Button
        self.esp_enable_button = QtWidgets.QPushButton("Enable ESP", self)
        visuals_layout.addWidget(self.esp_enable_button)
        
        # Distance Slider for ESP
        self.esp_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.esp_distance_slider.setRange(0, 500)
        visuals_layout.addWidget(self.esp_distance_slider)
        
        visuals_tab.setLayout(visuals_layout)
        self.tabs.addTab(visuals_tab, "Visuals")
        
        # Visuals AI Tab
        visuals_ai_tab = QtWidgets.QWidget()
        visuals_ai_layout = QtWidgets.QVBoxLayout()
        
        # AI ESP Button
        self.ai_esp_enable_button = QtWidgets.QPushButton("Enable AI ESP", self)
        visuals_ai_layout.addWidget(self.ai_esp_enable_button)
        
        # AI Distance Slider
        self.ai_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.ai_distance_slider.setRange(0, 500)
        visuals_ai_layout.addWidget(self.ai_distance_slider)
        
        visuals_ai_tab.setLayout(visuals_ai_layout)
        self.tabs.addTab(visuals_ai_tab, "Visuals AI")
        
        # Radar Tab
        radar_tab = QtWidgets.QWidget()
        radar_layout = QtWidgets.QVBoxLayout()
        
        # Radar Enable Button
        self.radar_enable_button = QtWidgets.QPushButton("Enable Radar", self)
        radar_layout.addWidget(self.radar_enable_button)
        
        # Radar Distance Slider
        self.radar_distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.radar_distance_slider.setRange(0, 500)
        radar_layout.addWidget(self.radar_distance_slider)
        
        radar_tab.setLayout(radar_layout)
        self.tabs.addTab(radar_tab, "Radar")

        # Display status label
        self.status_label = QtWidgets.QLabel("Status: Ready", self)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setGeometry(0, 550, 1000, 50)
        
        # Footer with branding
        self.footer_label = QtWidgets.QLabel("Made by: ocrzq (867x)", self)
        self.footer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.footer_label.setGeometry(0, 580, 1000, 50)
        self.footer_label.setStyleSheet("color: #00ffea; font-size: 14px;")
        
# ------------------------- Main Application -------------------------
def main():
    # Check hardware compatibility
    check_compatibility()

    cheat_thread = threading.Thread(target=cheat_loop, daemon=True)
    cheat_thread.start()
    hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
    hotkey_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    ui = CheatUI()
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()