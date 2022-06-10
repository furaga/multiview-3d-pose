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

from lib.MoveNet import MoveNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # For webcam input:
    all_cam_ids = [0, 2, 3, 4]
    all_caps = [cv2.VideoCapture(c) for c in all_cam_ids]
    
    movenet = MoveNet(args.model_select)

    finish = False
    prev = time.time()
    while not finish:
        all_kps = []
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
            img = movenet.draw_debug(img, dt, args.keypoint_score, keypoints, scores)
            all_imgs.append(img)

        show_img = cv2.vconcat(
            [
                cv2.hconcat(all_imgs[:2]),
                cv2.hconcat(all_imgs[2:]),
            ]
        )
        cv2.imshow(f"Cameras", show_img)
        if cv2.waitKey(1) == ord("q"):
            finish = True

    for cap in all_caps:
        cap.release()


if __name__ == "__main__":
    main()
