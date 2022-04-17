import argparse
from email.mime import image
from json import load
import numpy as np
import cv2
import glob
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt
import mediapipe as mp

import tools.geometry as geometry
import tools.calib.result as calib_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def create_plt(world_size=0.5):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Points")
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_zlim(-world_size, world_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax


def show_plt(block=True):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show(block=block)


def plot_points3d(ax, points3d, colors, center=None, s=8):
    points3d = np.array(points3d)
    colors = np.array(colors)
    mean = center if center is not None else np.mean(points3d, axis=0)
    x = points3d[:, 0] - mean[0]
    y = points3d[:, 1] - mean[1]
    z = points3d[:, 2] - mean[2]
    ax.scatter(x, y, z, c=colors, s=s)


def main(args):
    pose_dict = calib_result.pose_dict
    mtx = calib_result.camera_matrix

    all_video_paths = [args.video_dir / f"{name}.mp4" for name in pose_dict]
    for p in all_video_paths:
        assert p.exists()

    all_cam_ids = [p.stem for p in all_video_paths]
    all_caps = [cv2.VideoCapture(str(p)) for p in all_video_paths]

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose


    with mp_pose.Pose(
        model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        finish = False
        i_frame = 0
        while not finish:
            i_frame += 1
            is_draw = i_frame % 30 == 0
            if is_draw:
                ax = create_plt(world_size=10)
            all_kps = {}
            all_imgs = []
            for cam_id, cap in zip(all_cam_ids, all_caps):
                ret, img = cap.read()
                if not ret:
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)

                if results.pose_landmarks is not None:
                    kps = [
                        # x: px / width
                        # y: py / height
                        # z: 深度？平行投影で考えてる？今回は無視して良さそう
                        # visibility: score
                        (l.x, l.y, l.z, l.visibility)
                        for l in results.pose_landmarks.landmark
                    ]
                    kps = np.array(kps)
                    assert kps.shape == (33, 4), str(kps.shape)
                else:
                    kps = None
                all_kps[cam_id] = kps

                # # Draw the pose annotation on the image.
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                all_imgs.append(img)

            h, w = img.shape[:2]

            points3d = np.zeros((33, 3), float)
            for i in range(33):
                points2d = []
                projs = []
                for ci, cam_id in enumerate(all_cam_ids):
                    if all_kps[cam_id] is None:
                        continue
                    x, y, _, score = all_kps[cam_id][i]
                    if score < 0.9:
                        continue
                    kx = w * x
                    ky = h * y
                    points2d.append((kx, ky))
                    proj = np.matmul(mtx, pose_dict[cam_id][:3])
                    projs.append(proj)

                if len(points2d) >= 2:
                    pt3d = geometry.triangulate_nviews(
                        np.array(points2d), np.array(projs)
                    )
                    points3d[i] = pt3d

            if is_draw:
                plot_points3d(ax, points3d, (1, 0, 0))

            show_img = cv2.vconcat(
                [
                    cv2.hconcat(all_imgs[:2]),
                    cv2.hconcat(all_imgs[2:]),
                ]
            )
            cv2.imshow(f"Cameras", show_img)
            if cv2.waitKey(1) == ord("q"):
                finish = True
                
            if is_draw:
                show_plt(True)

        for cap in all_caps:
            cap.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
