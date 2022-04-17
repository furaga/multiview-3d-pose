import argparse
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import random
import random

import lib.geometry as geometry
import lib.visualize as visualize
import tools.calib.result as calib_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True, type=Path)
    parser.add_argument("--video_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    return args


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


# 0: 'left_shoulder',
# 1: 'right_shoulder',
# 2: 'left_elbow',
# 3: 'right_elbow',
# 4: 'left_wrist',
# 5: 'right_wrist',
# 6: 'left_hip',
# 7: 'right_hip',
# 8: 'left_knee',
# 9: 'right_knee',
# 10: 'left_ankle',
# 11: 'right_ankle',
# 12: 'top_head',
# 13: 'neck'

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


def draw_skeleton2d(img, kps, kps_colors):
    # Draw the pose annotation on the image.
    if kps is not None:
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
                    2,
                )
        for i, (x, y, score) in enumerate(kps):
            if score > args.threshold:
                cv2.circle(img, (int(x), int(y)), 6, kps_colors[i], -1)


def plot_skeleton3d(ax, points3d, kps_colors_plt):
    # keypoints
    visualize.plot_points3d(ax, points3d, kps_colors_plt, center=(0, 0, 0), s=64)

    # bones
    for a, b in skeleton:
        if np.any(points3d[a] != 0) and np.any(points3d[b] != 0):
            x1, y1, z1 = points3d[a]
            x2, y2, z2 = points3d[b]
            ax.plot(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                "o-",
                c=kps_colors_plt[a],
                ms=0,
                mew=1,
            )


def main(args):
    pose_dict = calib_result.pose_dict
    mtx = calib_result.camera_matrix
    distort = calib_result.distort

    all_video_paths = [args.video_dir / f"{name}.mp4" for name in pose_dict]
    for p in all_video_paths:
        assert p.exists(), str(p)

    all_cam_ids = [p.stem for p in all_video_paths]
    all_caps = [cv2.VideoCapture(str(p)) for p in all_video_paths]

    kps_dict = load_poses(args.sample_dir)

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

    rows = []
    while not finish:
        i_frame += 1
        is_draw3d = i_frame % 300 == 0

        # Images and Keypoints
        all_imgs = {}
        all_kps = {}
        for cam_id, cap in zip(all_cam_ids, all_caps):
            ret, img = cap.read()
            if not ret:
                continue
            all_imgs[cam_id] = img

            kps = None
            if i_frame - 1 in kps_dict[cam_id]:
                kps = kps_dict[cam_id][i_frame - 1]
            all_kps[cam_id] = kps

        if img is None:
            break

        h, w = img.shape[:2]

        #
        # Triangulate
        #
        points3d = np.zeros((14, 3), float)
        for i in range(14):
            points2d = []
            projs = []
            for cam_id in all_cam_ids:
                if all_kps[cam_id] is None:
                    continue
                x, y, score = all_kps[cam_id][i]
                if score <= args.threshold:
                    continue
                points2d.append((x, y))
                projs.append(np.matmul(mtx, pose_dict[cam_id][:3]))

            if len(points2d) >= 2:
                pts_undist = geometry.undistort(points2d, mtx, distort)
                ret, pt3d = geometry.triangulate_nviews_by2(pts_undist, np.array(projs))
                if ret:
                    points3d[i] = pt3d

        # Add Rows to Save
        row = [i_frame] + list(points3d.ravel())
        rows.append(row)

        # Draw 3D
        if is_draw3d:
            ax = visualize.create_plt(world_size=1)
            plot_skeleton3d(ax, points3d, kps_colors_plt)
            for cam_id, Rt in pose_dict.items():
                visualize.plot_camera(ax, Rt[:3, :3], Rt[:3, 3], cam_id, size=0.2)
            visualize.show_plt(True)

        # Draw 2D
        for cam_id, cap in zip(all_cam_ids, all_caps):
            draw_skeleton2d(all_imgs[cam_id], all_kps[cam_id], kps_colors)

        all_imgs_ls = [all_imgs[cam_id] for cam_id in all_cam_ids]
        show_img = cv2.vconcat(
            [
                cv2.hconcat(all_imgs_ls[:2]),
                cv2.hconcat(all_imgs_ls[2:]),
            ]
        )
        cv2.imshow(f"Cameras", show_img)
        if cv2.waitKey(1) == ord("q"):
            finish = True

    for cap in all_caps:
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv("keypoints3d.csv", header=None, index=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
