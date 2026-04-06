import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import requests
import csv
import re
from dotenv import load_dotenv
from image_processing import dip_algorithm_pro
from ocr_service import call_ocr_space
from config import *
import serial

load_dotenv()

# CẤU HÌNH & KẾT NỐI ARDUINO
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=0.1)
    print("Đã kết nối Arduino!")
except:
    arduino = None
    print("Không tìm thấy Arduino. Hệ thống sẽ chạy không có barrier.")

# Khởi tạo model YOLO
model = YOLO('best.pt') 
cap = cv2.VideoCapture(0)

def nothing(x): pass

# Khởi tạo Giao diện & Trackbars 
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_EXPANDED) 
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.createTrackbar("ROI_X", WINDOW_NAME, 150, 640, nothing)
cv2.createTrackbar("ROI_Y", WINDOW_NAME, 180, 480, nothing)
cv2.createTrackbar("ROI_W", WINDOW_NAME, 340, 640, nothing)
cv2.createTrackbar("ROI_H", WINDOW_NAME, 270, 480, nothing)
cv2.createTrackbar("CONF", WINDOW_NAME, 75, 100, nothing)

# BIẾN ĐIỀU KHIỂN LOGIC BARRIER
STABLE_DURATION = 2.0  # Thời gian đứng yên để nhận diện
is_tracking = False
start_time = 0
has_captured = False
detected_text = "NONE"
last_plate_raw = np.zeros((220, 320, 3), dtype=np.uint8)
last_plate_dip = np.zeros((220, 320, 3), dtype=np.uint8)

# Thêm biến kiểm soát trạng thái xe đi qua
barrier_status = "CLOSED"  # CLOSED, OPENING, WAITING_FOR_LEAVE
time_object_lost = 0
DELAY_CLOSE = 5.0  # Sau 5 giây không thấy biển số nữa thì đóng

print("Hệ thống khởi động thành công!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    tx = cv2.getTrackbarPos("ROI_X", WINDOW_NAME)
    ty = cv2.getTrackbarPos("ROI_Y", WINDOW_NAME)
    tw = cv2.getTrackbarPos("ROI_W", WINDOW_NAME)
    th = cv2.getTrackbarPos("ROI_H", WINDOW_NAME)
    conf_val = cv2.getTrackbarPos("CONF", WINDOW_NAME) / 100.0

    x_r, y_r = tx, ty
    x_r2, y_r2 = min(tx + tw, frame.shape[1]), min(ty + th, frame.shape[0])

    display_frame = frame.copy()
    current_time = time.time()
    
    cv2.rectangle(display_frame, (x_r, y_r), (x_r2, y_r2), (255, 165, 0), 2)
    
    roi_img = frame[y_r:y_r2, x_r:x_r2]
    results = model.predict(roi_img, conf=conf_val, verbose=False, imgsz=320)

    # NẾU NHÌN THẤY BIỂN SỐ
    if len(results[0].boxes) > 0:
        time_object_lost = 0  # Reset thời gian mất dấu
        
        box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        rx1, ry1, rx2, ry2 = x_r + bx1, y_r + by1, x_r + bx2, y_r + by2

        if barrier_status == "CLOSED":
            if not is_tracking:
                is_tracking = True
                start_time = current_time
            else:
                elapsed = current_time - start_time
                prog = min(elapsed / STABLE_DURATION, 1.0)
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.rectangle(display_frame, (x_r, y_r2 + 5), (x_r + int(prog*(x_r2-x_r)), y_r2 + 15), (0, 255, 255), -1)

                if elapsed >= STABLE_DURATION:
                    barrier_status = "OPENING"
                    plate_crop = frame[ry1:ry2, rx1:rx2]
                    
                    if plate_crop.size > 0:
                        processed = dip_algorithm_pro(plate_crop)
                        print("Gửi Cloud OCR...")
                        raw, clean = call_ocr_space(processed)
                        detected_text = clean
                        
                        # Điều khiển Barrier MỞ
                        if arduino and detected_text not in ["ERR", "TIMEOUT"]:
                            print(f"Ra lệnh: MỞ BARRIER cho xe {detected_text}")
                            # Gửi định dạng: O:BIENSO\n (Thêm \n để Arduino biết kết thúc chuỗi)
                            msg = f"O:{detected_text}\n"
                            arduino.write(msg.encode()) 
                            barrier_status = "WAITING_FOR_LEAVE"
                        
                        # Lưu dữ liệu
                        ts = time.strftime("%H%M%S")
                        img_path = f"{SAVE_DIR}/plate_{ts}.jpg"
                        cv2.imwrite(img_path, plate_crop)
                        with open(CSV_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([ts, raw, clean, img_path])
                        
                        last_plate_raw = cv2.resize(plate_crop, (320, 220))
                        last_plate_dip = cv2.cvtColor(cv2.resize(processed, (320, 220)), cv2.COLOR_GRAY2BGR)
                        
        elif barrier_status == "WAITING_FOR_LEAVE":
            # Xe vẫn đang đứng đó, giữ barrier mở
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)

    # NẾU KHÔNG CÒN THẤY BIỂN SỐ
    else:
        is_tracking = False
        
        # Nếu đang ở trạng thái đợi xe đi hẳn
        if barrier_status == "WAITING_FOR_LEAVE":
            if time_object_lost == 0:
                time_object_lost = current_time
            
            countdown = DELAY_CLOSE - (current_time - time_object_lost)
            print(f"Đang đếm ngược đóng barrier: {countdown:.1f}s")
            
            if countdown <= 0:
                if arduino:
                    print("Ra lệnh: ĐÓNG BARRIER")
                    # Gửi lệnh C\n để Arduino đóng servo và reset LCD
                    arduino.write(b"C\n") 
                barrier_status = "CLOSED"
                time_object_lost = 0
                detected_text = "NONE"

    # --- THIẾT KẾ CANVAS DASHBOARD ---
    C_W, C_H = 850, 520 
    canvas = np.zeros((C_H, C_W, 3), dtype=np.uint8)
    canvas[:] = (35, 35, 35) 

    cv2.putText(canvas, "LPR SYSTEM - HCMUTE", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    main_v = cv2.resize(display_frame, (500, 375))
    canvas[50:425, 20:520] = main_v
    cv2.rectangle(canvas, (20, 50), (520, 425), (100, 100, 100), 1)
    cv2.putText(canvas, "LIVE CAMERA", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    canvas[50:230, 550:830] = cv2.resize(last_plate_raw, (280, 180))
    cv2.rectangle(canvas, (550, 50), (830, 230), (0, 165, 255), 2)
    cv2.putText(canvas, "PLATE DETECTED", (550, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    canvas[245:425, 550:830] = cv2.resize(last_plate_dip, (280, 180))
    cv2.rectangle(canvas, (550, 245), (830, 425), (0, 255, 0), 2)
    cv2.putText(canvas, "DIP PROCESSED", (550, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.rectangle(canvas, (550, 440), (830, 490), (50, 50, 50), -1)
    cv2.putText(canvas, f"OCR: {detected_text}", (560, 475), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị trạng thái Barrier thực tế
    cv2.putText(canvas, f"BARRIER: {barrier_status}", (20, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, canvas)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()