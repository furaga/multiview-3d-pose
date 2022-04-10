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

import geometry
import calib.result
import calib.test_result


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

        # print("score", cnt)
        if cnt > best_score:
            best_score = cnt
            best_Rt = (R, t)

    if best_Rt is None:
        return False, None, None

    return True, best_Rt[0], best_Rt[1]


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


def draw_points3d(ax, points3d, colors, world_size=0.5, center=None, block=True, s=8):
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


def calc_colors(all_cam_ids):
    # cam_id_colors = {
    #     c: (
    #         random.random(),
    #         random.random(),
    #         random.random(),
    #     )
    #     for c in all_cam_ids
    # }

    cam_id_colors = {
        all_cam_ids[0]: (0, 0, 1),
        all_cam_ids[1]: (0, 1, 1),
        all_cam_ids[2]: (1, 0, 0),
        all_cam_ids[3]: (0.5, 0.5, 0.5),
    }

    return cam_id_colors


def load_correspondenses(board_dir, all_cam_ids):
    frame_to_dict = {}
    for name in all_cam_ids:
        df = pd.read_csv(board_dir / f"{name}.csv")
        for row in df.values:
            frame = int(row[0])
            pt2d = np.reshape([np.float32(t) for t in row[1:]], (-1, 1, 2))
            frame_to_dict.setdefault(frame, {})
            frame_to_dict[frame].setdefault(name, []).append(pt2d)

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
    ax = new_plt(world_size=1.5)
    for E, R, t, label in pose_infos:
        draw_camera(ax, R, t.ravel(), label)
    colors = [(0.2, 0.2, 0.2) for _ in points3d]
    draw_points3d(
        ax, points3d, colors, world_size=1.5, center=(0, 0, 0), block=True, s=8
    )
    show_plt(True)


def render_points2d(img, points2d, radius, color, thickness):
    for p in points2d.reshape((-1, 2)):
        x, y = p.astype(int)
        cv2.circle(img, (x, y), radius, color, thickness)


def main(args):
    debug_show = False
    args.out_dir.mkdir(exist_ok=True)

    w, h = 640, 480
    mtx = calib.result.camera_matrix
    distort = calib.result.distort
    all_ids = calib.result.all_ids
    all_cam_ids = sorted(set([c for _, c in all_ids]))
    cam_id_colors = calc_colors(all_cam_ids)

    # (cam_id1, cam_id2) -> (pts1, pts2)
    corr_dict = load_correspondenses(args.board_dir, all_cam_ids)

    # Calc relative camera poses
    pose_dict = {}
    for (cam_id1, cam_id2), (pts1, pts2) in corr_dict.items():
        # Undistortion
        pts1_undist = geometry.undistort(pts1, mtx, distort)
        pts2_undist = geometry.undistort(pts2, mtx, distort)

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
            pt3d = geometry.triangulate_nviews(
                np.array([p1, p2]), np.array([proj1, proj2])
            )
            points3d.append(pt3d)

        points3d = np.array(points3d)
        pose_dict[(cam_id1, cam_id2)] = R, t

        # visualize
        if debug_show:
            visualize_reconstruction(
                [
                    (np.eye(4), np.eye(3), np.zeros(3), cam_id1),
                    (E, R, t, cam_id2),
                ],
                points3d,
            )

            rvec, _ = cv2.Rodrigues(R)
            tvec = t.ravel()
            img = np.zeros((h, w * 2, 3), np.uint8)

            xshift = np.array([[w, 0]])
            render_points2d(img, pts1, 3, (255, 0, 0), -1)
            render_points2d(img, pts2 + xshift, 3, (255, 0, 0), -1)

            re_proj1, _ = cv2.projectPoints(
                points3d,
                np.zeros_like(rvec),
                np.zeros_like(tvec),
                mtx,
                distort,
            )
            render_points2d(img, re_proj1, 5, (0, 255, 0), 2)

            re_proj2, _ = cv2.projectPoints(points3d, rvec, tvec, mtx, distort)
            render_points2d(img, re_proj2 + xshift, 5, (0, 255, 0), 2)

            cv2.imshow("", img)
            cv2.waitKey(0)

    # print(list(edges.keys()))
    all_Rt = {}
    all_Rt[all_cam_ids[0]] = np.eye(4)

    ax = new_plt(world_size=1.5)
    draw_camera(ax, np.eye(3), np.zeros(3), all_cam_ids[0], size=0.5)
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

                draw_camera(ax, global_R, global_t, cam_id2, size=0.5)

        all_Rt[all_cam_ids[i]] = global_Rts[0] # np.mean(global_Rts, axis=0)

    print(all_Rt)
    
    print("KEYS", list(pose_dict.keys()))

    show_plt(True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
