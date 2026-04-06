import cv2
import os

def dip_algorithm_pro(img):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    return contrast

