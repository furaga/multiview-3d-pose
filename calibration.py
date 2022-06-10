# 外部パラメタをキャリブレーションする

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

import lib.geometry
import lib.visualize

gw, gh = 9, 7
# gw, gh = 10, 7
w, h = 640, 480
mtx = np.array(
    [
        [740, 0.0, 330],
        [0.0, 740, 221],
        [0.0, 0.0, 1.0],
    ]
)

distort = np.array(
    [
        0.059592372999535126,
        1.3343393909424992,
        -0.008491522926161756,
        0.006968330736260448,
        -5.81179797305232,
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
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
            points3d = lib.geometry.triangulate_nviews(
                np.array([p1, p2]), np.array([proj1, proj2])
            )
            if (
                lib.geometry.calc_depth(identity, points3d) > 0
                and lib.geometry.calc_depth(Rt, points3d) > 0
            ):
                cnt += 1

        # print("score", cnt)
        if cnt > best_score:
            best_score = cnt
            best_Rt = (R, t)

    if best_Rt is None:
        return False, None, None

    return True, best_Rt[0], best_Rt[1]


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


def plot_points3d(ax, points3d, colors, world_size=0.5, center=None, block=True, s=8):
    points3d = np.array(points3d)
    colors = np.array(colors)
    mean = center if center is not None else np.mean(points3d, axis=0)
    x = points3d[:, 0] - mean[0]
    y = points3d[:, 1] - mean[1]
    z = points3d[:, 2] - mean[2]
    ax.scatter(x, y, z, c=colors, s=s)


def make_view_position(R, t):
    p = -np.matmul(R.T, t.T).T
    return p


def make_Rt(R, t, expand=False):
    Rt = np.zeros((4 if expand else 3, 4), float)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()
    if expand:
        Rt[3, 3] = 1
    return Rt


def plot_camera(ax, R, t, id, size=1):
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


def calc_colors(all_cam_ids):
    cam_id_colors = {
        c: (
            random.random(),
            random.random(),
            random.random(),
        )
        for c in all_cam_ids
    }

    cam_id_colors[all_cam_ids[0]] = (0, 0, 1)
    cam_id_colors[all_cam_ids[1]] = (0, 1, 1)
    cam_id_colors[all_cam_ids[2]] = (1, 0, 0)
    cam_id_colors[all_cam_ids[3]] = (0.5, 0.5, 0.5)

    return cam_id_colors


def search_checkerboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    ret, corners = cv2.findChessboardCornersSB(gray, (gw, gh), None, flags)
    if ret:
        pt2d = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        pt2d = pt2d.reshape((-1, 1, 2))
        return pt2d

    return None


def calc_correspondenses(all_caps):
    frame_to_dict = {}
    frame = 0

    all_cam_counters = {k: 0 for k in all_caps.keys()}
    while True:
        frame += 1

        all_imgs = []
        boards = {}
        for cam_id, cap in all_caps.items():
            ret, img = cap.read()
            if not ret:
                all_imgs.append(img)
                continue
            pt2d = search_checkerboard(img)
            if pt2d is None:
                all_imgs.append(img)
                continue
            img = cv2.drawChessboardCorners(img, (gw, gh), pt2d, ret)
            boards.setdefault(cam_id, []).append(pt2d)
            all_imgs.append(img)

        show_img = cv2.vconcat(
            [
                cv2.hconcat(all_imgs[:2]),
                cv2.hconcat(all_imgs[2:]),
            ]
        )

        show_img = cv2.resize(show_img, None, fx=0.7, fy=0.7)
        cv2.imshow("show_img", show_img)
        if ord("q") == cv2.waitKey(1):
            exit(0)

        if len(boards) >= 2:
            if list(all_caps.keys())[0] in boards.keys():
                for cam_id, _ in boards.items():
                    all_cam_counters.setdefault(cam_id, 0)
                    all_cam_counters[cam_id] += 1
                frame_to_dict[frame] = boards

        enough = np.all([cnt >= 10 for _, cnt in all_cam_counters.items()])
        print(all_cam_counters)
        if enough:
            break

    corr_dict = {}
    for frame, dict in frame_to_dict.items():
        for cam_id1, pts1 in dict.items():
            for cam_id2, pts2 in dict.items():
                if cam_id1 >= cam_id2:
                    continue
                key = (cam_id1, cam_id2)
                corr_dict.setdefault(key, ([], []))
                corr_dict[key] = (
                    corr_dict[key][0] + pts1,
                    corr_dict[key][1] + pts2,
                )

    for k, (pts1, pts2) in corr_dict.items():
        pts1 = np.reshape(pts1, (-1, 2))
        pts2 = np.reshape(pts2, (-1, 2))
        corr_dict[k] = (pts1, pts2)

    return corr_dict


def visualize_reconstruction(pose_infos, points3d):
    ax = create_plt(world_size=1.5)
    for E, R, t, label in pose_infos:
        plot_camera(ax, R, t.ravel(), label)
    colors = [(0.2, 0.2, 0.2) for _ in points3d]
    plot_points3d(
        ax, points3d, colors, world_size=1.5, center=(0, 0, 0), block=True, s=8
    )
    show_plt(True)


def render_points2d(img, points2d, radius, color, thickness):
    for p in points2d.reshape((-1, 2)):
        x, y = p.astype(int)
        cv2.circle(img, (x, y), radius, color, thickness)


def calc_camera_poses(corr_dict, mtx, distort, all_cam_ids):
    pose_dict = {}
    for (cam_id1, cam_id2), (pts1, pts2) in corr_dict.items():
        # Undistortion
        pts1_undist = lib.geometry.undistort(pts1, mtx, distort)
        pts2_undist = lib.geometry.undistort(pts2, mtx, distort)

        # Find essential matrix
        E, inliers = cv2.findEssentialMat(pts1_undist, pts2_undist, mtx)

        # filter by inliers
        inliers = inliers.ravel()
        pts1_undist = [p for f, p in zip(inliers, pts1_undist) if f >= 1]
        pts2_undist = [p for f, p in zip(inliers, pts2_undist) if f >= 1]

        # Essential Matrix -> R, t
        ret, R, t = recoverPose(E, pts1_undist, pts2_undist, mtx)
        if not ret:
            print("Failed to recoverPose", (cam_id1, cam_id2))
            return

        proj1 = np.matmul(mtx, np.eye(4)[:3])
        proj2 = np.matmul(mtx, make_Rt(R, t))
        points3d = []
        for p1, p2 in zip(pts1_undist, pts2_undist):
            pt3d = lib.geometry.triangulate_nviews(
                np.array([p1, p2]), np.array([proj1, proj2])
            )
            points3d.append(pt3d)

        points3d = np.array(points3d)
        pose_dict[(cam_id1, cam_id2)] = R, t

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

            if key in pose_dict:
                M0 = all_Rt[all_cam_ids[0]]
                Mj = all_Rt[all_cam_ids[j]]
                base = np.matmul(M0, np.linalg.inv(Mj))

                R, t = pose_dict[key]
                Rt = make_Rt(R, t, expand=True)

                M = np.matmul(base, Rt)  # M0Mi
                global_R = M[:3, :3]
                global_t = M[:3, 3]
                global_Rts.append(make_Rt(global_R, global_t, expand=True))

        all_Rt[all_cam_ids[i]] = global_Rts[0]  # np.mean(global_Rts, axis=0)

    return all_Rt


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    all_cam_ids = [0, 2, 3, 4]
    # all_cam_ids = [
    #     "data/Recorded-Checkerboard/camera_0.mp4",
    #     "data/Recorded-Checkerboard/camera_2.mp4",
    #     "data/Recorded-Checkerboard/camera_3.mp4",
    #     "data/Recorded-Checkerboard/camera_4.mp4",
    # ]
    all_caps = {c: cv2.VideoCapture(c) for c in all_cam_ids}
    cam_id_colors = calc_colors(all_cam_ids)

    # (cam_id1, cam_id2) -> (pts1, pts2)
    corr_dict = calc_correspondenses(all_caps)

    # Calc relative camera poses
    all_Rt = calc_camera_poses(corr_dict, mtx, distort, all_cam_ids)

    print(all_Rt)

    # visualize
    ax = lib.visualize.create_plt(world_size=1)
    for cam_id, Rt in all_Rt.items():
        lib.visualize.plot_camera(ax, Rt[:3, :3], Rt[:3, 3], cam_id, size=0.2)
    lib.visualize.show_plt(True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
