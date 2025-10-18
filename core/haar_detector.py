# core/haar_detector.py
import cv2
import numpy as np
from typing import Tuple

def _load_haar():
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            return None
        return clf
    except Exception:
        return None

def detect_face_bbox(img) -> Tuple[Tuple[int,int,int,int], bool]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clf = _load_haar()
    if clf is not None:
        faces = clf.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) > 0:
            x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            return (x,y,w,h), True

    # Fallback: naive skin box
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = (0, 30, 50); upper1 = (25, 200, 255)
    lower2 = (160, 30, 50); upper2 = (179, 200, 255)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        H, W = img.shape[:2]
        return (int(W*0.2), int(H*0.2), int(W*0.6), int(H*0.6)), False
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return (x,y,w,h), False

def build_skin_mask(img, face_rect):
    x,y,w,h = face_rect
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower1 = (0, 15, 50); upper1 = (25, 180, 255)
    lower2 = (160, 15, 50); upper2 = (179, 180, 255)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    full = np.zeros(img.shape[:2], np.uint8)
    full[y:y+h, x:x+w] = mask
    return full

def eyes_lips_templates(face_rect, img_shape):
    x,y,w,h = face_rect
    H, W = img_shape[:2]
    eyes = np.zeros((H,W), np.uint8)
    lips = np.zeros((H,W), np.uint8)
    nl   = np.zeros((H,W), np.uint8)

    ey_y1 = y + int(0.25*h); ey_y2 = y + int(0.45*h)
    ey_x1 = x + int(0.15*w); ey_x2 = x + int(0.85*w)
    eyes[ey_y1:ey_y2, ey_x1:ey_x2] = 255

    lp_y1 = y + int(0.65*h); lp_y2 = y + int(0.80*h)
    lp_x1 = x + int(0.25*w); lp_x2 = x + int(0.75*w)
    lips[lp_y1:lp_y2, lp_x1:lp_x2] = 255

    nl_y1 = y + int(0.40*h); nl_y2 = y + int(0.70*h)
    nl_x1 = x + int(0.15*w); nl_x2 = x + int(0.85*w)
    nl[nl_y1:nl_y2, nl_x1:nl_x2] = 255

    return eyes, lips, nl
