import cv2
import os
def process_license_plate(img):
   
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Tăng cường độ tương phản (CLAHE - Cân bằng histogram cục bộ)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(blurred)
    
    # Thresholding
    # Otsu's method để tự động tìm ngưỡng tối ưu
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    return binary # Trả về ảnh đã được xử lý nhị phân
