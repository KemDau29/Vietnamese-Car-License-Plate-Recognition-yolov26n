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
    print("Da ket noi Arduino!")
except:
    arduino = None
    print("Khong tim thay Arduino. He thong se chay khong co barrier.")

model = YOLO('best.pt')

ip_address = "214.200.40.229"
video_url  = f"http://{ip_address}:8080/video"
cap        = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong the ket noi voi camera dien thoai!")

def nothing(x): pass

# CONFIG WINDOW  
CONFIG_WIN     = "CONFIG PANEL"
config_visible = False
_roi_cache     = [150, 180, 340, 270, 75]   # X Y W H CONF%

def show_config_window():
    cv2.namedWindow(CONFIG_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG_WIN, 420, 220)
    cv2.createTrackbar("ROI_X",  CONFIG_WIN, _roi_cache[0], 640, nothing)
    cv2.createTrackbar("ROI_Y",  CONFIG_WIN, _roi_cache[1], 480, nothing)
    cv2.createTrackbar("ROI_W",  CONFIG_WIN, _roi_cache[2], 640, nothing)
    cv2.createTrackbar("ROI_H",  CONFIG_WIN, _roi_cache[3], 480, nothing)
    cv2.createTrackbar("CONF %", CONFIG_WIN, _roi_cache[4], 100, nothing)

def hide_config_window():
    try:
        _roi_cache[0] = cv2.getTrackbarPos("ROI_X",  CONFIG_WIN)
        _roi_cache[1] = cv2.getTrackbarPos("ROI_Y",  CONFIG_WIN)
        _roi_cache[2] = cv2.getTrackbarPos("ROI_W",  CONFIG_WIN)
        _roi_cache[3] = cv2.getTrackbarPos("ROI_H",  CONFIG_WIN)
        _roi_cache[4] = cv2.getTrackbarPos("CONF %", CONFIG_WIN)
    except: pass
    try: cv2.destroyWindow(CONFIG_WIN)
    except: pass

# Khởi tạo trackbar ẩn để cache giá trị
show_config_window()
hide_config_window()


C_BG       = ( 8,  18,  14)
C_PANEL    = (12,  26,  20)
C_BORDER   = (35,  65,  50)
C_ACCENT   = ( 0, 210, 140)   
C_ACCENT2  = ( 0, 170, 255)   
C_WARN     = ( 0,  80, 220)   
C_TEXT     = (200, 230, 215)
C_SUBTEXT  = ( 80, 120,  95)
C_GREEN    = ( 40, 230, 100)
C_YELLOW   = (  0, 210, 230)
C_DIV      = ( 22,  44,  33)

F  = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

def T(img, s, x, y, col=C_TEXT, sc=0.42, th=1, f=F):
    cv2.putText(img, str(s), (x, y), f, sc, col, th, AA)

def Tc(img, s, cx, y, col=C_TEXT, sc=0.42, th=1, f=F):
    w = cv2.getTextSize(str(s), f, sc, th)[0][0]
    cv2.putText(img, str(s), (cx - w//2, y), f, sc, col, th, AA)

def corners(img, x1, y1, x2, y2, col, sz=14, th=2):
    for ax, ay, dx, dy in [(x1,y1,sz,sz),(x2,y1,-sz,sz),(x1,y2,sz,-sz),(x2,y2,-sz,-sz)]:
        cv2.line(img,(ax,ay),(ax+dx,ay),   col,th,AA)
        cv2.line(img,(ax,ay),(ax,ay+dy),   col,th,AA)

def hline(img, x1, x2, y, col=C_DIV):
    cv2.line(img,(x1,y),(x2,y),col,1,AA)

def vline(img, x, y1, y2, col=C_DIV):
    cv2.line(img,(x,y1),(x,y2),col,1,AA)

def pbar(img, x1, y, x2, p, h=5, bg=C_BORDER, fg=C_ACCENT):
    cv2.rectangle(img,(x1,y),(x2,y+h),bg,-1)
    fw = int((x2-x1)*max(0.0,min(1.0,p)))
    if fw>0: cv2.rectangle(img,(x1,y),(x1+fw,y+h),fg,-1)

def led(img, cx, cy, col, off=False):
    ring = C_SUBTEXT if off else col
    core = C_DIV if off else tuple(min(255,int(c*1.4)) for c in col)
    cv2.circle(img,(cx,cy),7,ring,-1,AA)
    cv2.circle(img,(cx,cy),4,core,-1,AA)
    cv2.circle(img,(cx,cy),7,ring,1,AA)

def rect_fill_alpha(img, x1, y1, x2, y2, col, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)


STABLE_DURATION  = 2.0
is_tracking      = False
start_time       = 0
detected_text    = "NONE"
last_plate_raw   = np.zeros((1,1,3), dtype=np.uint8)
last_plate_dip   = np.zeros((1,1,3), dtype=np.uint8)
barrier_status   = "CLOSED"
time_object_lost = 0
DELAY_CLOSE      = 5.0
frame_count      = 0
log_lines        = []   

def add_log(msg):
    ts = time.strftime("%H:%M:%S")
    log_lines.append(f"{ts}  {msg}")
    if len(log_lines) > 5:
        log_lines.pop(0)


C_W, C_H = 1020, 650

# Các vùng (tính từ trên xuống)
HEADER_H  = 54          # tên thành viên
TITLE_H   = 40          # thanh hệ thống
TOP_H     = HEADER_H + TITLE_H          # = 94
BOTTOM_H  = 118                         # status bar dưới
MID_Y1    = TOP_H + 12
MID_Y2    = C_H - BOTTOM_H - 10
MID_H     = MID_Y2 - MID_Y1

# Cột trái/phải
CAM_X1   = 10
CAM_X2   = 660
RIGHT_X1 = 674
RIGHT_X2 = C_W - 10

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow(WINDOW_NAME, C_W, C_H)
print("He thong khoi dong!  [C] Config  [Q] Quit")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_count  += 1
    current_time  = time.time()

    # trackbar 
    if config_visible:
        tx       = cv2.getTrackbarPos("ROI_X",  CONFIG_WIN)
        ty       = cv2.getTrackbarPos("ROI_Y",  CONFIG_WIN)
        tw_      = cv2.getTrackbarPos("ROI_W",  CONFIG_WIN)
        th_      = cv2.getTrackbarPos("ROI_H",  CONFIG_WIN)
        conf_val = cv2.getTrackbarPos("CONF %", CONFIG_WIN) / 100.0
        _roi_cache[:] = [tx, ty, tw_, th_, int(conf_val*100)]
    else:
        tx, ty, tw_, th_ = _roi_cache[:4]
        conf_val = _roi_cache[4] / 100.0

    x_r,  y_r  = tx, ty
    x_r2, y_r2 = min(tx+tw_, frame.shape[1]), min(ty+th_, frame.shape[0])

    display_frame = frame.copy()
    cv2.rectangle(display_frame,(x_r,y_r),(x_r2,y_r2),C_ACCENT2,2,AA)
    corners(display_frame,x_r,y_r,x_r2,y_r2,C_ACCENT2,sz=18,th=2)

    roi_img = frame[y_r:y_r2, x_r:x_r2]
    results  = model.predict(roi_img, conf=conf_val, verbose=False, imgsz=320)

    # LOGIC BARRIER
    plate_detected = len(results[0].boxes) > 0

    if plate_detected:
        time_object_lost = 0
        box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        bx1,by1,bx2,by2 = map(int, box.xyxy[0])
        rx1,ry1,rx2,ry2 = x_r+bx1, y_r+by1, x_r+bx2, y_r+by2

        if barrier_status == "CLOSED":
            if not is_tracking:
                is_tracking = True
                start_time  = current_time
            else:
                elapsed = current_time - start_time
                prog    = min(elapsed / STABLE_DURATION, 1.0)
                cv2.rectangle(display_frame,(rx1,ry1),(rx2,ry2),C_YELLOW,2,AA)
                corners(display_frame,rx1,ry1,rx2,ry2,C_YELLOW,sz=10)

                if elapsed >= STABLE_DURATION:
                    barrier_status = "OPENING"
                    plate_crop = frame[ry1:ry2, rx1:rx2]

                    if plate_crop.size > 0:
                        processed = dip_algorithm_pro(plate_crop)
                        add_log("Sending to Cloud OCR...")
                        raw, clean = call_ocr_space(processed)
                        detected_text = clean

                        if arduino and detected_text not in ["ERR","TIMEOUT"]:
                            add_log(f"OPEN BARRIER -> {detected_text}")
                            arduino.write(f"O:{detected_text}\n".encode())
                            barrier_status = "WAITING_FOR_LEAVE"

                        ts_str   = time.strftime("%H%M%S")
                        img_path = f"{SAVE_DIR}/plate_{ts_str}.jpg"
                        cv2.imwrite(img_path, plate_crop)
                        with open(CSV_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([ts_str, raw, clean, img_path])

                        last_plate_raw = plate_crop.copy()
                        last_plate_dip = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        elif barrier_status == "WAITING_FOR_LEAVE":
            cv2.rectangle(display_frame,(rx1,ry1),(rx2,ry2),C_GREEN,2,AA)
            corners(display_frame,rx1,ry1,rx2,ry2,C_GREEN,sz=10)
    else:
        is_tracking = False
        if barrier_status == "WAITING_FOR_LEAVE":
            if time_object_lost == 0:
                time_object_lost = current_time
            countdown = DELAY_CLOSE - (current_time - time_object_lost)
            if frame_count % 15 == 0:
                add_log(f"Closing barrier in {countdown:.1f}s ...")
            if countdown <= 0:
                if arduino:
                    add_log("BARRIER CLOSED")
                    arduino.write(b"C\n")
                barrier_status   = "CLOSED"
                time_object_lost = 0
                detected_text    = "NONE"

    canvas = np.full((C_H, C_W, 3), C_BG, dtype=np.uint8)

    barrier_col = {
        "CLOSED":            C_SUBTEXT,
        "OPENING":           C_YELLOW,
        "WAITING_FOR_LEAVE": C_GREEN,
    }.get(barrier_status, C_SUBTEXT)

    ard_col  = C_GREEN if arduino else C_WARN
    ard_text = "CONNECTED" if arduino else "OFFLINE"

    # HEADER - THÔNG TIN
    rect_fill_alpha(canvas, 0, 0, C_W, HEADER_H, C_PANEL, alpha=1.0)
    # accent line dưới header
    cv2.rectangle(canvas,(0,HEADER_H),(C_W,HEADER_H+2),C_ACCENT,-1)

    T(canvas,"Vu Hoang Nam  23110042",  14, 22, C_ACCENT, sc=0.54, th=1)
    T(canvas,"Nguyen Tien Cuong  23110007", 14, 46, C_ACCENT, sc=0.54, th=1)

    vline(canvas, 320, 6, HEADER_H-4, C_BORDER)

    T(canvas,"LICENSE PLATE RECOGNITION - Final Project", 334, 22, C_SUBTEXT, sc=0.50, th=1)
    T(canvas,"Digital image processing - Lect: Dr. Hoang Van Dung ",  334, 44, C_SUBTEXT, sc=0.42, th=1)

    ts_str = time.strftime("%H:%M:%S    %d/%m/%Y")
    tw_px  = cv2.getTextSize(ts_str, F, 0.47, 1)[0][0]
    T(canvas, ts_str, C_W-tw_px-14, 24, C_TEXT,    sc=0.47, th=1)
    hint = "[C] Config   [Q] Quit"
    hw   = cv2.getTextSize(hint, F, 0.37, 1)[0][0]
    T(canvas, hint,   C_W-hw-14,    44, C_SUBTEXT, sc=0.37, th=1)

    # TITLE BAR
    rect_fill_alpha(canvas, 0, HEADER_H+2, C_W, TOP_H, (10,22,17), alpha=1.0)
    hline(canvas, 0, C_W, TOP_H, C_BORDER)

    cv2.putText(canvas,"HCMUTE",(14,HEADER_H+32),FD,1.05,C_ACCENT,2,AA)

    T(canvas,f"BARRIER: {barrier_status.replace('_',' ')}",
      168, HEADER_H+24, barrier_col, sc=0.48, th=1)
    T(canvas,f"ARDUINO: {ard_text}",
      168, HEADER_H+40, ard_col, sc=0.40, th=1)

    # LIVE CAMERA 
    cam_w = CAM_X2 - CAM_X1 - 2
    cam_h = MID_Y2 - MID_Y1 - 40  

    cam_r = cv2.resize(display_frame, (cam_w, cam_h))
    canvas[MID_Y1+1 : MID_Y1+1+cam_h, CAM_X1+1 : CAM_X2-1] = cam_r

    # Vẽ khung viền bao quanh toàn bộ khu vực (bao gồm cả chỗ chứa text/bar)
    cv2.rectangle(canvas, (CAM_X1, MID_Y1), (CAM_X2, MID_Y2), C_ACCENT2, 1, AA)
    corners(canvas, CAM_X1, MID_Y1, CAM_X2, MID_Y2, C_ACCENT2, sz=18, th=2)

    T(canvas, "LIVE FEED", CAM_X1+6, MID_Y1-5, C_ACCENT2, sc=0.38)
    T(canvas, f"CONF {int(conf_val*100)}%", CAM_X2-76, MID_Y1+cam_h-7, C_SUBTEXT, sc=0.37)

    #Progress Bar
    if is_tracking and barrier_status == "CLOSED":
        elapsed = current_time - start_time
        prog = min(elapsed / STABLE_DURATION, 1.0)
        # Tọa độ Y bây giờ nằm TRONG khoảng từ MID_Y1+cam_h đến MID_Y2
        T(canvas, f"LOCKING {prog*100:.0f}%", CAM_X1+6, MID_Y2-22, C_YELLOW, sc=0.40)
        pbar(canvas, CAM_X1+5, MID_Y2-12, CAM_X2-5, prog, h=6, fg=C_YELLOW)
    else:
        lc = C_GREEN if barrier_status == "WAITING_FOR_LEAVE" else C_ACCENT
        lt = "BARRIER OPEN - VEHICLE PASSING" if barrier_status == "WAITING_FOR_LEAVE" else "SYSTEM READY"
        T(canvas, lt, CAM_X1+6, MID_Y2-22, lc, sc=0.40)
        pv = 1.0 if barrier_status != "CLOSED" else 0.0
        fg_ = C_GREEN if barrier_status == "WAITING_FOR_LEAVE" else C_DIV
        pbar(canvas, CAM_X1+5, MID_Y2-12, CAM_X2-5, pv, h=6, fg=fg_)

    # PANEL
    GAP   = 14
    P_MID = MID_Y1 + MID_H // 2

    # Panel 1 — PLATE DETECTED
    P1Y1, P1Y2 = MID_Y1, P_MID - GAP//2
    ph1 = P1Y2 - P1Y1 - 2
    pw1 = RIGHT_X2 - RIGHT_X1 - 2
    if last_plate_raw.size > 1:
        canvas[P1Y1+1:P1Y2-1, RIGHT_X1+1:RIGHT_X2-1] = cv2.resize(last_plate_raw,(pw1,ph1))
    cv2.rectangle(canvas,(RIGHT_X1,P1Y1),(RIGHT_X2,P1Y2),C_ACCENT2,1,AA)
    corners(canvas,RIGHT_X1,P1Y1,RIGHT_X2,P1Y2,C_ACCENT2,sz=12,th=2)
    T(canvas,"PLATE DETECTED",RIGHT_X1+6,P1Y1-5,C_ACCENT2,sc=0.38)

    # Panel 2 — DIP PROCESSED
    P2Y1, P2Y2 = P_MID + GAP//2, MID_Y2
    ph2 = P2Y2 - P2Y1 - 2
    pw2 = RIGHT_X2 - RIGHT_X1 - 2
    if last_plate_dip.size > 1:
        canvas[P2Y1+1:P2Y2-1, RIGHT_X1+1:RIGHT_X2-1] = cv2.resize(last_plate_dip,(pw2,ph2))
    cv2.rectangle(canvas,(RIGHT_X1,P2Y1),(RIGHT_X2,P2Y2),C_ACCENT,1,AA)
    corners(canvas,RIGHT_X1,P2Y1,RIGHT_X2,P2Y2,C_ACCENT,sz=12,th=2)
    T(canvas,"DIP PROCESSED",RIGHT_X1+6,P2Y1-5,C_ACCENT,sc=0.38)

    # BOTTOM STATUS BAR 
    BAR_Y = C_H - BOTTOM_H
    rect_fill_alpha(canvas, 0, BAR_Y, C_W, C_H, C_PANEL, alpha=1.0)
    cv2.rectangle(canvas,(0,BAR_Y),(C_W,BAR_Y+2),C_ACCENT,-1)   # accent top line

    CX = [10, 240, 460, 670]   # x bắt đầu 4 cột

    # Col 1 — OCR
    T(canvas,"OCR RESULT", CX[0], BAR_Y+22, C_SUBTEXT, sc=0.40)
    ocr_col = C_GREEN if detected_text not in ["NONE","ERR","TIMEOUT"] else C_SUBTEXT
    cv2.putText(canvas, detected_text, (CX[0], BAR_Y+72),
                FD, 1.2, ocr_col, 2, AA)
    vline(canvas, CX[1]-14, BAR_Y+8, C_H-8, C_BORDER)

    # Col 2 — BARRIER
    T(canvas,"BARRIER", CX[1], BAR_Y+22, C_SUBTEXT, sc=0.40)
    led(canvas, CX[1]+7, BAR_Y+44, barrier_col, off=(barrier_status=="CLOSED"))
    T(canvas, barrier_status.replace("_"," "), CX[1]+20, BAR_Y+50,
      barrier_col, sc=0.44, th=1)
    if barrier_status=="WAITING_FOR_LEAVE" and time_object_lost>0:
        cd = DELAY_CLOSE-(current_time-time_object_lost)
        T(canvas,f"Close in {cd:.1f}s",CX[1],BAR_Y+72,C_YELLOW,sc=0.38)
    vline(canvas, CX[2]-14, BAR_Y+8, C_H-8, C_BORDER)

    # Col 3 — ARDUINO
    T(canvas,"ARDUINO", CX[2], BAR_Y+22, C_SUBTEXT, sc=0.40)
    led(canvas, CX[2]+7, BAR_Y+44, ard_col, off=(not arduino))
    T(canvas, ard_text, CX[2]+20, BAR_Y+50, ard_col, sc=0.44, th=1)
    vline(canvas, CX[3]-14, BAR_Y+8, C_H-8, C_BORDER)

    # Col 4 — EVENT LOG (ASCII only)
    T(canvas,"EVENT LOG", CX[3], BAR_Y+22, C_SUBTEXT, sc=0.40)
    for i, line in enumerate(log_lines[-4:]):
        alpha   = 0.35 + 0.65*((i+1)/4)
        log_col = tuple(int(c*alpha) for c in C_TEXT)
        T(canvas, line, CX[3], BAR_Y+38+i*20, log_col, sc=0.37)

    cv2.imshow(WINDOW_NAME, canvas)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        config_visible = not config_visible
        if config_visible:
            show_config_window()
            add_log("Config panel opened")
        else:
            hide_config_window()
            add_log("Config panel closed")

cap.release()
cv2.destroyAllWindows()