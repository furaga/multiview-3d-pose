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


def bgr2hsv(bgr):
    bgr = np.reshape(bgr, (1, 1, 3)).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return hsv.ravel()


def hsv2bgr(hsv):
    hsv = np.reshape(hsv, (1, 1, 3)).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr.ravel()


def extract_color(img, col1, col2):
    lower = np.min([col1, col2], axis=0)
    upper = np.max([col1, col2], axis=0)
    mask = cv2.inRange(img, lower, upper)
    return mask


def load_color_range(color_name, low_thr, high_thr):
    print("Load color", color_name)
    colors = []

    all_img_paths = Path(f"../sample/marker/{color_name}").glob("*.png")
    for img_path in all_img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(w, h) / 2 * 0.8
        for y in range(h):
            for x in range(w):
                dist_sq = (cx - x) ** 2 + (cy - y) ** 2
                if dist_sq <= radius * radius:
                    colors.append(bgr2hsv(img[y, x]))

    colors = np.array(colors)
    lower = np.percentile(colors, low_thr, axis=0).astype(np.uint8)
    higher = np.percentile(colors, high_thr, axis=0).astype(np.uint8)
    print(color_name, lower, higher)

    return lower, higher


def find_marker(hsv_img, lower, higher):
    h, w = hsv_img.shape[:2]
    img_area = h * w

    m = extract_color(hsv_img, lower, higher)
    _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best = None
    for c in contours:
        ca = cv2.contourArea(c)
        if ca > img_area * 0.01:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > hsv_img.shape[0] * 0.05:
            continue

        if radius <= 2:
            continue

        ratio = ca / (np.pi * radius * radius)
        if ratio < 0.4:
            continue

        if best is None:
            best = ca, center, radius

        if best[0] < ca:
            best = ca, center, radius

    if best is not None:
        ca, center, radius = best
        return True, center, radius, m

    return False, None, None, m


def tile_images(imgs):
    return cv2.vconcat(
        [
            cv2.hconcat(imgs[:2]),
            cv2.hconcat(imgs[2:]),
        ]
    )


class MarkerTracker:
    def __init__(self, cam_id, marker_id):
        self.cam_id = cam_id
        self.marker_id = marker_id
        self.N_BUFFER = 32
        self.history = []
        self.is_tracking = False
        self.last_update_time = 0
        self.center = None
        self.radius = None

    def update(self, ret, center, radius, cur_time):
        self.history.append((ret, np.array(center), radius, cur_time))

        if len(self.history) > self.N_BUFFER:
            self.history = self.history[-self.N_BUFFER :]
        assert len(self.history) <= self.N_BUFFER

        self.update_tracking_state(cur_time)
        self.tracking(cur_time)

        return self.is_tracking, self.center, self.radius

    def update_tracking_state(self, cur_time):
        if self.is_tracking:
            # N秒以上更新されなければ
            n = 1.0
            if cur_time - self.last_update_time > n:
                self.is_tracking = False
                print("Finish Tracking", self.cam_id, self.marker_id)
        else:
            # Nフレームcenterがだいたい同じ場所にいたら追跡開始
            n = 30
            thr = 20
            if len(self.history) >= n:
                ret_all = np.all([h[0] for h in self.history[-n:]], axis=0)
                if not ret_all:
                    return
                cmin = np.min([h[1] for h in self.history[-n:]], axis=0)
                cmax = np.max([h[1] for h in self.history[-n:]], axis=0)
                d = cmax - cmin
                if np.max(d) < thr:
                    self.is_tracking = True
                    self.center = self.history[-1][1]
                    self.radius = self.history[-1][2]
                    self.last_update_time = cur_time
                    print("Start Tracking", self.cam_id, self.marker_id)

    def tracking(self, cur_time):
        if not self.is_tracking:
            return

        ret, center, radius, cur_time = self.history[-1]
        if not ret:
            return

        thr = 100
        dist = np.linalg.norm(self.center - center)
        if dist > thr:
            return

        self.last_update_time = cur_time

        # 適当にフィルターする？
        self.center = center
        self.radius = radius


def main(args):
    marker_colors = [
        load_color_range("moss_green", 10, 95),
        load_color_range("yellow_green", 15, 85),
        load_color_range("purple", 10, 90),
        load_color_range("green", 10, 90),
        # load_color_range("blue", 10, 90),
        load_color_range("orange", 14, 92),
    ]

    all_caps = [cv2.VideoCapture(i) for i in [0, 2, 3, 4]]

    trackers = {
        cam_id: [MarkerTracker(cam_id, mi) for mi, _ in enumerate(marker_colors)]
        for cam_id, _ in enumerate(all_caps)
    }
    cur_time = 0

    while True:
        imgs = []
        masks = []
        raw_masks = []
        cur_time += 1 / 30
        for cam_id, cap in enumerate(all_caps):
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.blur(img, (4, 4))
            mask = np.zeros_like(img)
            raw_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            for mi, (lower, higher) in enumerate(marker_colors):
                ret, center, radius, m = find_marker(hsv_img, lower, higher)
                raw_mask += m

                # Tracking
                tr = trackers[cam_id][mi]
                ret_tr, center_tr, radius_tr = tr.update(ret, center, radius, cur_time)

                c = hsv2bgr(higher)
                c = (int(c[2]), int(c[1]), int(c[0]))
                if ret_tr:
                    mask = cv2.circle(mask, center_tr, radius_tr, c, -1)
                elif ret:
                    mask = cv2.circle(mask, center, radius, c, 2)

            imgs.append(img)
            masks.append(mask)
            raw_masks.append(raw_mask)

        cv2.imshow("img", tile_images(imgs))
        cv2.imshow("mask", tile_images(masks))
        cv2.imshow("raw_masks", tile_images(raw_masks))
        if ord("q") == cv2.waitKey(1):
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
