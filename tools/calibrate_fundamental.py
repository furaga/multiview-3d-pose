import argparse
from email.mime import image
import numpy as np
import cv2
import glob
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt

import geometry
import calib.result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def recoverPose(E, points1, points2, mtx):

    R1, R2, t_ = cv2.decomposeEssentialMat(E)

    Rs = [R1, R2, R1, R2]
    ts = [t_, t_, -t_, -t_]

    best_score = 0
    best_Rt = None
    for R, t in zip(Rs, ts):
        Rt = np.zeros((3, 4), float)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.ravel()

        identity = np.eye(4)[:3]
        proj1 = np.matmul(mtx, identity)
        proj2 = np.matmul(mtx, Rt)

        cnt = 0
        for p1, p2 in zip(points1, points2):
            points3d = geometry.triangulate_nviews(
                np.array([p1, p2]), np.array([proj1, proj2])
            )
            if (
                geometry.calc_depth(identity, points3d) > 0
                and geometry.calc_depth(Rt, points3d) > 0
            ):
                cnt += 1

        if cnt > best_score:
            best_score = cnt
            best_Rt = (R, t)

    if best_Rt is None:
        return False, None, None

    return True, best_Rt[0], best_Rt[1]


def draw_points3d(points3d, colors, world_size=0.5, center=None, block=True, s=8):
    points3d = np.array(points3d)
    colors = np.array(colors)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    mean = center if center is not None else np.mean(points3d, axis=0)
    x = points3d[:, 0] - mean[0]
    y = points3d[:, 1] - mean[1]
    z = points3d[:, 2] - mean[2]

    ax.scatter(x, y, z, c=colors, s=s)
    ax.set_title("3D Points")

    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_zlim(-world_size, world_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show(block=block)


def make_view_position(R, t):
    p = -np.matmul(R, t.T).T
    return p


def make_Rt(R, t, expand=False):
    Rt = np.zeros((4 if expand else 3, 4), float)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()
    if expand:
        Rt[3, 3] = 1
    return Rt


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    all_ids = calib.result.all_ids
    all_cam_ids = sorted(set([c for _, c in all_ids]))
    cam_id_colors = {
        c: (
            random.random(),
            random.random(),
            random.random(),
        )
        for c in all_cam_ids
    }

    # frame -> Dict[str, List[Point]]
    frame_to_dict = {}
    for name in all_cam_ids:
        df = pd.read_csv(args.board_dir / f"{name}.csv")
        for row in df.values:
            frame = int(row[0])
            pt2d = np.reshape([np.float32(t) for t in row[1:]], (-1, 1, 2))
            frame_to_dict.setdefault(frame, {})
            frame_to_dict[frame].setdefault(name, []).append(pt2d)

    corr = {}
    for frame, dict in frame_to_dict.items():
        for cam_id1, pts1 in dict.items():
            for cam_id2, pts2 in dict.items():
                if cam_id1 >= cam_id2:
                    continue
                key = (cam_id1, cam_id2)
                corr.setdefault(key, ([], []))
                corr[key] = (
                    corr[key][0] + pts1,
                    corr[key][1] + pts2,
                )

    mtx = calib.result.camera_matrix
    distort = calib.result.distort
    Rt_dict = {}
    for (cam_id1, cam_id2), (pts1, pts2) in corr.items():
        pts1 = np.reshape(pts1, (-1, 1, 2))
        pts2 = np.reshape(pts2, (-1, 1, 2))
        pts1_undist = geometry.undistort(np.array(pts1), mtx, distort)
        pts2_undist = geometry.undistort(np.array(pts2), mtx, distort)

        E, inliers = cv2.findEssentialMat(pts1_undist, pts2_undist, mtx)
        inliers = inliers.ravel()
        pts1_undist = [p for f, p in zip(inliers, pts1_undist) if f >= 1]
        pts2_undist = [p for f, p in zip(inliers, pts2_undist) if f >= 1]

        # print(inliers)
        ret, R, t = recoverPose(E, pts1_undist, pts2_undist, mtx)
        if not ret:
            print("failed", (cam_id1, cam_id2))

        Rt_dict[(cam_id1, cam_id2)] = R, t

    #
    # Visualize
    #
    points3d = [np.zeros(3, float)]
    colors = [cam_id_colors[all_cam_ids[0]]]

    # print(list(edges.keys()))
    all_Rt = {}
    all_Rt[all_cam_ids[0]] = np.eye(4)

    for i in range(1, len(all_cam_ids)):
        cam_id2 = all_cam_ids[i]
        global_Rts = []
        for j in range(i):
            if j != 0:
                continue
            cam_id1 = all_cam_ids[j]
            key = cam_id1, cam_id2

            if key in Rt_dict:
                M0 = all_Rt[all_cam_ids[0]]
                Mj = all_Rt[all_cam_ids[j]]
                base = np.matmul(M0, np.linalg.inv(Mj))

                R, t = Rt_dict[key]
                Rt = make_Rt(R, t, expand=True)

                M = np.matmul(base, Rt)  # M0Mi
                global_R = M[:3, :3]
                global_t = M[:3, 3]

                pt3d = make_view_position(global_R, global_t)
                points3d.append(pt3d)
                colors.append(cam_id_colors[all_cam_ids[i]])
                global_Rts.append(make_Rt(global_R, global_t, expand=True))

        all_Rt[all_cam_ids[i]] = np.mean(global_Rts, axis=0)

    print(points3d)
    draw_points3d(points3d, colors, world_size=1.5, center=(0, 0, 0), block=True, s=64)


if __name__ == "__main__":
    args = parse_args()
    main(args)
