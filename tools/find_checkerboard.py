import argparse
import numpy as np
import cv2
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    all_video_paths = args.input_dir.glob("*.mp4")
    all_caps = {p.stem: cv2.VideoCapture(str(p)) for p in all_video_paths}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    finish = False
    frame = 0
    while not finish:
        if frame % 120 == 0:
            print("Frame:", frame)
        for name, cap in all_caps.items():
            for _ in range(3):
                ret, img = cap.read()
            if not ret:
                finish = True
                break

            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gw, gh = 10, 7
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE

            ret, corners = cv2.findChessboardCornersSB(gray, (gw, gh), None, flags)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                text = ",".join([str(v) for v in corners2.ravel()])
                with open(args.out_dir / f"{name}.csv", "a") as f:
                    f.write(str(frame) + "," + text + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
