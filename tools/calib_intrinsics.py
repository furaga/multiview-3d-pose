import argparse
from email.mime import image
import numpy as np
import cv2
import glob
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--board_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_pt3d = np.zeros((10 * 7, 3), np.float32)

    # 1辺2cm
    gw, gh = 10, 7
    obj_pt3d[:, :2] = np.mgrid[0:gw, 0:gh].T.reshape(-1, 2)
    obj_pt3d[:, :2] *= 0.02

    all_video_paths = args.input_dir.glob("*.mp4")
    all_caps = {p.stem: cv2.VideoCapture(str(p)) for p in all_video_paths}

    frame = 0
    all_dfs = {}
    for name in all_caps:
        df = pd.read_csv(args.board_dir / f"{name}.csv")
        f2c = {}
        for row in df.values:
            f2c[int(row[0])] = np.reshape([np.float32(t) for t in row[1:]], (-1, 1, 2))
        all_dfs[name] = f2c

    all_ids = []
    all_pts3d = []
    all_pts2d = []
    while True:
        images = []
        frame += 1
        finish = False
        if frame % 50 == 0:
            print("Frame", frame)

        for name, cap in all_caps.items():
            ret, img = cap.read()
            if not ret:
                finish = True
                break

            h, w = img.shape[:2]

            if frame in all_dfs[name]:
                corners2 = all_dfs[name][frame]
                img = cv2.drawChessboardCorners(img, (gw, gh), corners2, ret)
                all_ids.append((frame, name))
                all_pts3d.append(obj_pt3d)
                all_pts2d.append(corners2)

            images.append(img)

        if finish:
            break

    print("Calibrating: # of data =", len(all_pts3d))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_pts3d, all_pts2d, (w, h), None, None
    )

    if ret:
        with open(args.out_dir / f"intrinsics_{name}.txt", "w") as f:
            f.write(",".join([str(v) for v in mtx.ravel()]))
            f.write("\n")
            f.write(",".join([str(v) for v in dist.ravel()]))
            f.write("\n")

    print(repr(np.array([v.ravel() for v in rvecs])))
    print(repr(np.array([v.ravel() for v in tvecs])))
    print(all_ids)


if __name__ == "__main__":
    args = parse_args()
    main(args)
