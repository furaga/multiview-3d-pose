import numpy as np
import cv2
from pathlib import Path


def extract_color(img, col1, col2):
    lower = np.min([col1, col2], axis=0)
    upper = np.max([col1, col2], axis=0)
    mask = cv2.inRange(img, lower, upper)
    return mask


def bgr2hsv(bgr):
    bgr = np.reshape(bgr, (1, 1, 3)).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv.ravel()


def hsv2bgr(hsv):
    hsv = np.reshape(hsv, (1, 1, 3)).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr.ravel()


def load_color_range(color_name):
    print("Load color", color_name)
    colors = []
    for img_path in Path("sample/marker").glob(f"{color_name}*.png"):
        img = cv2.imread(str(img_path))
        colors += [
            bgr2hsv(c) for c in img.reshape((-1, 3)) if np.any(c != np.zeros_like(c))
        ]
    colors = np.array(colors)
    lower = np.percentile(colors, 5, axis=0).astype(np.uint8)
    higher = np.percentile(colors, 95, axis=0).astype(np.uint8)
    return lower, higher


def find_marker(img, lower, higher):
    m = extract_color(img, lower, higher)
    _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 0:
        return False, None, None
    contour_areas = [cv2.contourArea(c) for c in contours]
    largest = contours[np.argmax(contour_areas)]
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return True, (int(x), int(y)), int(radius)
