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
    parser.add_argument("--sample_dir", required=True, type=Path)
    parser.add_argument("--video_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    return args


def new_plt(world_size=0.5):
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


def draw_points3d(ax, points3d, colors, center=None, s=8):
    points3d = np.array(points3d)
    colors = np.array(colors)
    mean = center if center is not None else np.mean(points3d, axis=0)
    x = points3d[:, 0] - mean[0]
    y = points3d[:, 1] - mean[1]
    z = points3d[:, 2] - mean[2]
    ax.scatter(x, y, z, c=colors, s=s)
    print(x, y, z, mean)
    print("===")
    return mean


def load_poses(json_dir):
    import json

    all_json_paths = json_dir.glob("*.json")
    kps_dict = {}
    for p in all_json_paths:
        kps_dict.setdefault(p.stem, {})
        with open(p) as f:
            people = json.loads(f.read())
            for person in people:
                kps = person["keypoints"]
                kps = np.reshape(kps, (-1, 3))
                kps_dict[p.stem][person["i_frame"]] = kps
    return kps_dict


def make_view_position(R, t):
    p = -np.matmul(R.T, t.T).T
    return p


def draw_camera(ax, R, t, id, size=1):
    x = [size, 0, 0]
    y = [0, size, 0]
    z = [0, 0, size]
    c = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(3):
        p0 = make_view_position(R, t)
        p = np.array([x[i], y[i], z[i]])
        p = np.matmul(R.T, p.T).T + p0
        ax.plot(
            [p0[0], p[0]], [p0[1], p[1]], [p0[2], p[2]], "o-", c=c[i], ms=4, mew=0.5
        )
        ax.text(p0[0], p0[1], p0[2], id, None)


skeleton = [
    [0, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    [0, 13],
    [1, 13],
    [12, 13],
    [0, 6],
    [1, 7],
    [6, 7],
    [6, 8],
    [8, 10],
    [7, 9],
    [9, 11],
]


def main(args):
    pose_dict = calib_result.pose_dict
    mtx = calib_result.camera_matrix
    distort = calib_result.distort

    all_video_paths = [args.video_dir / f"{name}.mp4" for name in pose_dict]
    for p in all_video_paths:
        assert p.exists()

    all_cam_ids = [p.stem for p in all_video_paths]
    all_caps = [cv2.VideoCapture(str(p)) for p in all_video_paths]

    kps_dict = load_poses(args.sample_dir)

    import random

    kps_colors = [
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for _ in range(14)
    ]
    kps_colors_plt = [(r / 255, g / 255, b / 255) for r, g, b in kps_colors]

    finish = False
    i_frame = 0
    while not finish:
        i_frame += 1
        is_draw = i_frame % 1 == 0
        if is_draw:
            ax = new_plt(world_size=1)

        all_kps = {}
        all_imgs = []
        for cam_id, cap in zip(all_cam_ids, all_caps):
            ret, img = cap.read()
            if not ret:
                continue

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            kps = None
            if i_frame - 1 in kps_dict[cam_id]:
                kps = kps_dict[cam_id][i_frame - 1]
            all_kps[cam_id] = kps

            # Draw the pose annotation on the image.
            img.flags.writeable = True
            if kps is not None:
                for i, (x, y, score) in enumerate(kps):
                    if score > args.threshold:
                        cv2.circle(img, (int(x), int(y)), 3, kps_colors[i], -1)

                for a, b in skeleton:
                    if a >= 14 or b >= 14:
                        continue
                    if kps[a][2] > args.threshold and kps[b][2] > args.threshold:
                        x1, y1 = kps[a][:2]
                        x2, y2 = kps[b][:2]
                        cv2.line(
                            img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            kps_colors[a],
                            3,
                        )

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            all_imgs.append(img)

        h, w = img.shape[:2]

        points3d = np.zeros((14, 3), float)
        for i in range(14):
            points2d = []
            projs = []
            for ci, cam_id in enumerate(all_cam_ids):
                if all_kps[cam_id] is None:
                    continue
                x, y, score = all_kps[cam_id][i]
                if score <= args.threshold:
                    continue
                kx = x
                ky = y
                points2d.append((kx, ky))
                proj = np.matmul(mtx, pose_dict[cam_id][:3])
                projs.append(proj)

            if len(points2d) >= 4:
                pts_undist = geometry.undistort(points2d, mtx, distort)
                pt3d = geometry.triangulate_nviews(np.array(pts_undist), np.array(projs))
                points3d[i] = pt3d

        if is_draw:
            draw_center = draw_points3d(ax, points3d, kps_colors_plt, s=32)

            for a, b in skeleton:
                if a >= 14 or b >= 14:
                    continue
                if (
                    all_kps[cam_id][a][2] > args.threshold
                    and all_kps[cam_id][b][2] > args.threshold
                ):
                    x1, y1, z1 = points3d[a]
                    x1 -= draw_center[0]
                    y1 -= draw_center[1]
                    z1 -= draw_center[2]
                    x2, y2, z2 = points3d[b]
                    x2 -= draw_center[0]
                    y2 -= draw_center[1]
                    z2 -= draw_center[2]
                    print([x1, x2], [y1, y2], [z1, z2])
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        [z1, z2],
                        "o-",
                        c=kps_colors_plt[a],
                        ms=0,
                        mew=2,
                    )

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
            for cam_id, Rt in pose_dict.items():
                draw_camera(ax, Rt[:3, :3], Rt[:3, 3], cam_id, size=0.2)

            show_plt(True)

    for cap in all_caps:
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
