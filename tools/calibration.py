import argparse
from email.mime import image
import numpy as np
import cv2
import glob
from pathlib import Path


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

    # Arrays to store object points and image points from all the images.
    objpoints = {}  # 3d point in real world space
    imgpoints = {}  # 2d points in image plane.

    all_video_paths = args.input_dir.glob("*.mp4")
    all_caps = {p.stem: cv2.VideoCapture(str(p)) for p in all_video_paths}

    frame = 0

    out_video = None
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

            board_path = args.board_dir / f"{name}_{frame:05d}.txt"
            if board_path.exists():
                with open(board_path) as f:
                    corners2 = [np.float32(t) for t in f.read().strip().split(",")]
                    corners2 = np.reshape(corners2, (-1, 1, 2))
                img = cv2.drawChessboardCorners(img, (gw, gh), corners2, ret)
                objpoints.setdefault("all", []).append(obj_pt3d)
                imgpoints.setdefault("all", []).append(corners2)

            images.append(img)

        if finish:
            break

        black = np.zeros_like(img)
        images.append(black)

        assert len(images) == 6
        combined = cv2.vconcat(
            [
                cv2.hconcat(images[0:2]),
                cv2.hconcat(images[2:4]),
                cv2.hconcat(images[4:6]),
            ]
        )

        if out_video is None:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out_video = cv2.VideoWriter(
                str(args.out_dir / "calibration.mp4"),
                fourcc,
                25,
                (combined.shape[1], combined.shape[0]),
            )

        out_video.write(combined)
        cv2.imshow("combined", cv2.resize(combined, None, fx=0.5, fy=0.5))
        if ord("q") == cv2.waitKey(1):
            break

    print("Calibrating...")
    all_pts3d = []
    all_pts2d = []
    for name in objpoints.keys():
        all_pts3d += objpoints[name]
        all_pts2d += imgpoints[name]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_pts3d, all_pts2d, (w, h), None, None
    )
    if ret:
        with open(args.out_dir / f"intrinsics_{name}.txt", "w") as f:
            f.write(",".join([str(v) for v in mtx.ravel()]))
            f.write("\n")
            f.write(",".join([str(v) for v in dist.ravel()]))
            f.write("\n")

    if out_video is not None:
        out_video.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
