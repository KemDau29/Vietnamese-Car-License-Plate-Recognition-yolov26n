import requests
import re
import cv2
from dotenv import load_dotenv
from config import API_KEY
load_dotenv()

def call_ocr_space(img_np):
    try:
        _, img_encoded = cv2.imencode('.jpg', img_np)
        img_bytes = img_encoded.tobytes()
        
        payload = {
            'apikey': API_KEY,
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 2 
        }
        files = {'filename.jpg': img_bytes}
        
        response = requests.post('https://api.ocr.space/parse/image', files=files, data=payload, timeout=10)
        result = response.json()
        
        if result.get('OCRExitCode') == 1:
            raw_text = result['ParsedResults'][0]['ParsedText']
            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            return raw_text, clean_text
        return "ERR", "ERR"
    except:
        return "TIMEOUT", "TIMEOUT"