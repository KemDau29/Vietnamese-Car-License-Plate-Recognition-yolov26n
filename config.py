import os
from dotenv import load_dotenv
load_dotenv()

# API
API_KEY = os.getenv("OCR_API_KEY", "")

# Camera & Window
WINDOW_NAME = "LPR Dashboard HCMUTE"
SAVE_DIR = "lpr_output"
CSV_FILE = os.path.join(SAVE_DIR, "lpr_log.csv")

# Hardware
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600

# Logic
STABLE_DURATION = 2.0
DELAY_CLOSE = 5.0

# UI Colors (BGR)
COLOR_ORANGE = (255, 165, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_DARK = (35, 35, 35)