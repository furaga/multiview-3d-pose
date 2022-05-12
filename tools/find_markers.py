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
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
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


def load_color_range(color_name):
    print("Load color", color_name)
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


def find_marker(hsv_img, lower, higher):
    h, w = hsv_img.shape[:2]
    img_area = h * w

    m = extract_color(hsv_img, lower, higher)
    _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.imshow("m", m)

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
        if ratio < 0.5:
            continue

        if best is None:
            best = ca, center, radius

        if best[0] < ca:
            best = ca, center, radius

    if best is not None:
        ca, center, radius = best
        return True, center, radius

    return False, None, None


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

    def update(self, ret, center, radius, cur_time):
        self.history.append((ret, center, radius, cur_time))

        if len(self.history) > self.N_BUFFER:
            self.history = self.history[-self.N_BUFFER :]
        assert len(self.history) <= self.N_BUFFER

        self.update_tracking_state(cur_time)
        self.tracking(cur_time)

        return ret, center, radius

    def update_tracking_state(self, cur_time):
        if self.is_tracking:
            # N秒以上更新されなければ
            n = 1.0
            if cur_time - self.last_update_time > n:
                self.is_tracking = False
                print("Finish Tracking", self.cam_id, self.marker_id)
        else:
            # Nフレームcenterがだいたい同じ場所にいたら追跡開始
            n = 10
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
                    self.last_update_time = cur_time
                    print("Start Tracking", self.cam_id, self.marker_id)

    def tracking(self, cur_time):
        pass



def main(args):
    marker_colors = [
        #        load_color_range("moss_green"),
        load_color_range("yellow_green"),
        # load_color_range("purple"),
        # load_color_range("green"),
        # load_color_range("blue"),
        # load_color_range("orange"),
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
        cur_time += 1 / 30
        for cam_id, cap in enumerate(all_caps):
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.blur(img, (4, 4))
            mask = np.zeros_like(img)

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for mi, (lower, higher) in enumerate(marker_colors):
                ret, center, radius = find_marker(hsv_img, lower, higher)

                # Tracking
                tr = trackers[cam_id][mi]
                ret, center, radius = tr.update(ret, center, radius, cur_time)

                c = hsv2bgr(higher)
                c = (int(c[0]), int(c[1]), int(c[2]))
                if ret:
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
