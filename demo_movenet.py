import socketserver
import cv2
import numpy as np
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
from pathlib import Path

from lib.MoveNet import MoveNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # For webcam input:
    if args.video_dir is None:
        all_cam_ids = [0, 2, 3, 4]
    else:
        all_video_paths = [
            args.video_dir / "camera_0.mp4",
            args.video_dir / "camera_2.mp4",
            args.video_dir / "camera_3.mp4",
            args.video_dir / "camera_4.mp4",
        ]
        for p in all_video_paths:
            assert p.exists(), str(p)

    all_cam_ids = [p.stem for p in all_video_paths]
    print(all_cam_ids)

    all_caps = [cv2.VideoCapture(str(c)) for c in all_video_paths]

    movenet = MoveNet(args.model_select)

    finish = False
    prev = time.time()

    with open("data/movenet_keypoints.csv", "w", encoding="utf8") as f:
        i_frame = 0
        while not finish:
            all_imgs = []
            dt = time.time() - prev
            prev = time.time()
            for cam_id, cap in zip(all_cam_ids, all_caps):
                if not cap.isOpened():
                    continue

                ret, img = cap.read()
                if not ret:
                    continue

                keypoints, scores = movenet.run_inference(img)
                img = movenet.draw_debug(
                    img, dt, args.keypoint_score, keypoints, scores
                )
                all_imgs.append(img)

                f.write(str(cam_id) + "," + str(i_frame) + ",")
                for kp, s in zip(keypoints, scores):
                    f.write(str(kp[0]) + "," + str(kp[1]) + "," + str(s) + ",")
                f.write("\n")

            show_img = cv2.vconcat(
                [
                    cv2.hconcat(all_imgs[:2]),
                    cv2.hconcat(all_imgs[2:]),
                ]
            )
            cv2.imshow(f"Cameras", show_img)
            if cv2.waitKey(1) == ord("q"):
                finish = True

            i_frame += 1

    for cap in all_caps:
        cap.release()


if __name__ == "__main__":
    main()
