import cv2
from ultralytics import YOLO
import time
import os

# ==========================================
# 1. CẤU HÌNH CƠ BẢN
# ==========================================
model = YOLO('best.pt')  # Nạp mô hình của bạn
conf_threshold = 0.6    # Độ tự tin tối thiểu (60%) để "chốt" ảnh

# Thư mục lưu ảnh biển số đã crop để kiểm tra
save_dir = "captured_plates"
if not os.path.exists(save_dir): os.makedirs(save_dir)

# ==========================================
# 2. KHỞI TẠO CAMERA & CẤU HÌNH ROI
# ==========================================
cap = cv2.VideoCapture(0)
# Cố định độ phân giải để tọa độ ROI chính xác
FRAME_W, FRAME_H = 640, 480
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

# ĐỊNH NGHĨA VÙNG ROI (Vùng xe sẽ dừng trước barrier)
# Dạng: (x_start, y_start, x_end, y_end) - Tính theo pixel
# Bạn hãy căn chỉnh 4 số này sao cho nó nằm ở CHÍNH GIỮA hoặc DƯỚI CÙNG khung hình
roi_box = (150, 200, 490, 450) # Ví dụ một hình chữ nhật lớn ở giữa dưới

# Biến logic điều khiển
last_capture_time = 0
COOLDOWN_TIME = 5 # (giây) - Tránh "chốt" ảnh liên tục cho cùng một xe

print("Hệ thống Barrier ảo đang chạy... Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Tạo một bản sao để vẽ hiển thị, không làm bẩn ảnh gốc đưa vào YOLO
    display_frame = frame.copy()
    current_time = time.time()

    # VẼ VÙNG ROI LÊN MÀN HÌNH ---
    # Vẽ khung nét đứt hoặc màu khác để người lái xe (hoặc bạn khi demo) căn chỉnh
    x_r, y_r, x_r2, y_r2 = roi_box
    cv2.rectangle(display_frame, (x_r, y_r), (x_r2, y_r2), (255, 0, 0), 2) # Màu xanh dương
    cv2.putText(display_frame, "VUNG NHAN DIEN (ROI)", (x_r, y_r - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # CẮT VÙNG ẢNH ROI ĐỂ ĐƯA VÀO YOLO
    roi_img = frame[y_r:y_r2, x_r:x_r2]

    # CHỈ CHẠY YOLO TRÊN ROI ---
    if roi_img.size > 0:
        # Tăng hiệu năng bằng cách ép imgsz nhỏ hơn (ví dụ 320 hoặc 640 tùy model)
        results = model.predict(roi_img, conf=conf_threshold, verbose=False, imgsz=320)

        # Biến tạm để tìm biển số tốt nhất trong frame
        best_plate_in_frame = None
        max_conf = 0

        # --- D. XỬ LÝ KẾT QUẢ ĐẦU RA CỦA YOLO ---
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                # Tìm biển số có độ tự tin cao nhất trong frame này
                if conf > max_conf:
                    max_conf = conf
                    best_plate_in_frame = box

        # CHỐT ẢNH 1 LẦN
        # Chỉ chốt ảnh khi: 
        # 1. Tìm thấy biển số 
        # 2. Độ tự tin cao (> conf_threshold) 
        # 3. Đã qua thời gian chờ (Cooldown)
        if best_plate_in_frame is not None and (current_time - last_capture_time > COOLDOWN_TIME):
            
            # 1. Lấy tọa độ BIỂN SỐ so với vùng ROI
            x1, y1, x2, y2 = map(int, best_plate_in_frame.xyxy[0])
            
            # 2. CROP BIỂN SỐ TỪ ẢNH GỐC (Frame)
            # Quan trọng: Phải cộng thêm tọa độ gốc của ROI (x_r, y_r)
            real_x1 = x_r + x1
            real_y1 = y_r + y1
            real_x2 = x_r + x2
            real_y2 = y_r + y2
            
            # Đảm bảo không cắt ra ngoài khung ảnh gốc
            real_x1 = max(0, real_x1); real_y1 = max(0, real_y1)
            real_x2 = min(FRAME_W, real_x2); real_y2 = min(FRAME_H, real_y2)

            final_plate_crop = frame[real_y1:real_y2, real_x1:real_x2]

            if final_plate_crop.size > 0:
                # XỬ LÝ CHỐT ẢNH & LƯU
                last_capture_time = current_time # Cập nhật thời gian chốt
                
                # Hiển thị ảnh chốt được
                cv2.imshow("== CHOT ANH BIEN SO ==", final_plate_crop)
                
                # Lưu ảnh làm dữ liệu 
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/plate_{timestamp}_{max_conf:.2f}.jpg"
                cv2.imwrite(filename, final_plate_crop)
                
                print(f"-> Đã chốt ảnh: {filename}. Đang mở barrier ảo...")

            # 4. VẼ KHUNG LÊN MÀN HÌNH HIỂN THỊ (Sử dụng tọa độ thật)
            cv2.rectangle(display_frame, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Bien So: {max_conf:.2f}", (real_x1, real_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị luồng video chính
    cv2.imshow("Hệ thống Barrier ảo (ROI)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()