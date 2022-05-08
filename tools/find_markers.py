import argparse
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import random


def parse_args():
    parser = argparse.ArgumentParser()
    #    parser.add_argument("--video_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def extract_color(img, col1, col2):
    lower = np.min([col1, col2], axis=0)
    upper = np.max([col1, col2], axis=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # print(lower, upper)
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
    colors = []
    for img_path in Path("../sample/marker").glob(f"{color_name}*.png"):
        img = cv2.imread(str(img_path))
        colors += [
            bgr2hsv(c) for c in img.reshape((-1, 3)) if np.any(c != np.zeros_like(c))
        ]
    colors = np.array(colors)
    lower = np.percentile(colors, 5, axis=0).astype(np.uint8)
    higher = np.percentile(colors, 95, axis=0).astype(np.uint8)
    return lower, higher


def tile_images(imgs):
    return cv2.vconcat(
        [
            cv2.hconcat(imgs[:2]),
            cv2.hconcat(imgs[2:]),
        ]
    )


def main(args):
    blue = load_color_range("blue")
    orange = load_color_range("orange")
    marker_colors = [blue, orange]

    all_caps = [cv2.VideoCapture(i) for i in [0, 2, 3, 4]]
    while True:
        imgs = []
        masks = []
        for cam_id, cap in enumerate(all_caps):
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.blur(img, (4, 4))

            mask = np.zeros_like(img)
            for lower, higher in marker_colors:
                m = extract_color(img, lower, higher)
                _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if len(contours) <= 0:
                    continue
                contour_areas = [cv2.contourArea(c) for c in contours]
                largest = contours[np.argmax(contour_areas)]

                (x, y), radius = cv2.minEnclosingCircle(largest)
                center = (int(x), int(y))
                radius = int(radius)
                c = hsv2bgr(higher)
                c = (int(c[0]), int(c[1]), int(c[2]))
                mask = cv2.circle(mask, center, radius, c, -1)

            imgs.append(img)
            masks.append(mask)

        cv2.imshow("img", tile_images(imgs))
        cv2.imshow("mask", tile_images(masks))
        if ord("q") == cv2.waitKey(1):
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
