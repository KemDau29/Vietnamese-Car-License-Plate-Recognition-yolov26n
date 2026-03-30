import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import requests
import csv
import re
from dotenv import load_dotenv
load_dotenv()


# 1. CẤU HÌNH & THIẾT LẬP BAN ĐẦU
API_KEY = os.getenv("OCR_API_KEY", "")
if not API_KEY:
    raise ValueError("OCR_API_KEY missing!")

WINDOW_NAME = "LPR Dashboard HCMUTE"
SAVE_DIR = "lpr_output"
CSV_FILE = os.path.join(SAVE_DIR, "lpr_log.csv")

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Full_Text', 'Clean_Plate', 'Image_Path'])

# Khởi tạo model YOLO
model = YOLO('best.pt') 
cap = cv2.VideoCapture(0)

# CÁC HÀM XỬ LÝ OCR & DIP
def nothing(x): pass

def call_ocr_space(img_np):
    """ Gửi ảnh qua API OCR.Space """
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

def dip_algorithm_pro(img):
    """ Tiền xử lý ảnh cho OCR """
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Tăng độ tương phản để chữ rõ nét hơn trên Cloud
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    return contrast

# Khởi tạo Giao diện & Trackbars 
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_EXPANDED) 
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.createTrackbar("ROI_X", WINDOW_NAME, 150, 640, nothing)
cv2.createTrackbar("ROI_Y", WINDOW_NAME, 180, 480, nothing)
cv2.createTrackbar("ROI_W", WINDOW_NAME, 340, 640, nothing)
cv2.createTrackbar("ROI_H", WINDOW_NAME, 270, 480, nothing)
cv2.createTrackbar("CONF", WINDOW_NAME, 75, 100, nothing)

# Biến điều khiển
STABLE_DURATION = 2.0
is_tracking = False
start_time = 0
has_captured = False
detected_text = "NONE"
last_plate_raw = np.zeros((220, 320, 3), dtype=np.uint8)
last_plate_dip = np.zeros((220, 320, 3), dtype=np.uint8)

print("Hệ thống khởi động thành công!")

# LUỒNG XỬ LÝ CHÍNH
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Lấy thông số từ Trackbars
    tx = cv2.getTrackbarPos("ROI_X", WINDOW_NAME)
    ty = cv2.getTrackbarPos("ROI_Y", WINDOW_NAME)
    tw = cv2.getTrackbarPos("ROI_W", WINDOW_NAME)
    th = cv2.getTrackbarPos("ROI_H", WINDOW_NAME)
    conf_val = cv2.getTrackbarPos("CONF", WINDOW_NAME) / 100.0

    x_r, y_r = tx, ty
    x_r2, y_r2 = min(tx + tw, frame.shape[1]), min(ty + th, frame.shape[0])

    display_frame = frame.copy()
    current_time = time.time()
    
    # Vẽ vùng ROI
    cv2.rectangle(display_frame, (x_r, y_r), (x_r2, y_r2), (255, 165, 0), 2)
    
    # YOLO nhận diện trong ROI
    roi_img = frame[y_r:y_r2, x_r:x_r2]
    results = model.predict(roi_img, conf=conf_val, verbose=False, imgsz=320)

    if len(results[0].boxes) > 0:
        box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        rx1, ry1, rx2, ry2 = x_r + bx1, y_r + by1, x_r + bx2, y_r + by2

        if not has_captured:
            if not is_tracking:
                is_tracking = True
                start_time = current_time
            else:
                elapsed = current_time - start_time
                # Vẽ Progress Bar
                prog = min(elapsed / STABLE_DURATION, 1.0)
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.rectangle(display_frame, (x_r, y_r2 + 5), (x_r + int(prog*(x_r2-x_r)), y_r2 + 15), (0, 255, 255), -1)

                if elapsed >= STABLE_DURATION:
                    has_captured = True
                    plate_crop = frame[ry1:ry2, rx1:rx2]
                    if plate_crop.size > 0:
                        processed = dip_algorithm_pro(plate_crop)
                        
                        # GỌI API OCR
                        print("Gửi Cloud OCR...")
                        raw, clean = call_ocr_space(processed)
                        detected_text = clean
                        
                        # Lưu dữ liệu
                        ts = time.strftime("%H%M%S")
                        img_path = f"{SAVE_DIR}/plate_{ts}.jpg"
                        cv2.imwrite(img_path, plate_crop)
                        with open(CSV_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([ts, raw, clean, img_path])
                        
                        last_plate_raw = cv2.resize(plate_crop, (320, 220))
                        last_plate_dip = cv2.cvtColor(cv2.resize(processed, (320, 220)), cv2.COLOR_GRAY2BGR)
        else:
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
    else:
        is_tracking, start_time, has_captured = False, 0, False

    # --- THIẾT KẾ CANVAS DASHBOARD ---
    # Giảm kích thước tổng để không bị mất phần dưới màn hình
    C_W, C_H = 850, 520 
    canvas = np.zeros((C_H, C_W, 3), dtype=np.uint8)
    canvas[:] = (35, 35, 35) 

    # 1. Tiêu đề 
    cv2.putText(canvas, "LPR SYSTEM - HCMUTE", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # 2. Khung Camera Chính 
    main_v = cv2.resize(display_frame, (500, 375))
    canvas[50:425, 20:520] = main_v
    cv2.rectangle(canvas, (20, 50), (520, 425), (100, 100, 100), 1)
    cv2.putText(canvas, "LIVE CAMERA", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 3. Khung Ảnh Crop 
    canvas[50:230, 550:830] = cv2.resize(last_plate_raw, (280, 180))
    cv2.rectangle(canvas, (550, 50), (830, 230), (0, 165, 255), 2)
    cv2.putText(canvas, "PLATE DETECTED", (550, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # 4. Khung Ảnh DIP 
    canvas[245:425, 550:830] = cv2.resize(last_plate_dip, (280, 180))
    cv2.rectangle(canvas, (550, 245), (830, 425), (0, 255, 0), 2)
    cv2.putText(canvas, "DIP PROCESSED", (550, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 5. Hiển thị Kết quả OCR 
    cv2.rectangle(canvas, (550, 440), (830, 490), (50, 50, 50), -1)
    cv2.putText(canvas, f"OCR: {detected_text}", (560, 475), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # 6. Trạng thái 
    status_msg = "SCANNING..." if is_tracking else "READY"
    if has_captured: status_msg = "COMPLETED"
    cv2.putText(canvas, f"STATUS: {status_msg}", (20, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, canvas)
   
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()