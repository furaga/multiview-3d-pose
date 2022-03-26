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


def show_plt(block):
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
    p = -np.matmul(R, t.T).T
    return p


def make_Rt(R, t, expand=False):
    Rt = np.zeros((4 if expand else 3, 4), float)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()
    if expand:
        Rt[3, 3] = 1
    return Rt


def draw_camera(ax, R, t, id):
    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_title("3D Points")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # world_size = 0.6
    # ax.set_xlim(-world_size, world_size)
    # ax.set_ylim(-world_size, world_size)
    # ax.set_zlim(-world_size, world_size)

    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]
    c = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(3):
        p0 = make_view_position(R, t)
        p = np.array([x[i], y[i], z[i]])
        p = np.matmul(R, p.T).T + p0
        ax.plot(
            [p0[0], p[0]], [p0[1], p[1]], [p0[2], p[2]], "o-", c=c[i], ms=4, mew=0.5
        )

    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show(block=True)


def main(args):
    # draw_camera(
    #     np.matmul(calib.test_result.Ry(np.pi / 4), calib.test_result.Rx(np.pi / 4)),
    #     np.array([0.5, 0.2, 0.3], float),
    #     "test"
    # )
    # exit(0)
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
    cam_id_colors = {
        all_cam_ids[0]: (0, 0, 1),
        all_cam_ids[1]: (0, 1, 1),
        all_cam_ids[2]: (1, 0, 0),
        all_cam_ids[3]: (0.5, 0.5, 0.5),
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
                if key not in corr:
                    print(key, frame)
                    corr.setdefault(key, ([], []))
                    corr[key] = (
                        corr[key][0] + pts1,
                        corr[key][1] + pts2,
                    )
    w, h = 640, 480


    mtx = calib.result.camera_matrix
    distort = calib.result.distort
    Rt_dict = {}
    for (cam_id1, cam_id2), (pts1, pts2) in corr.items():
        pts1 = np.reshape(pts1, (-1, 2))
        pts2 = np.reshape(pts2, (-1, 2))
        pts1_undist = geometry.undistort(pts1, mtx, distort)
        pts2_undist = geometry.undistort(pts2, mtx, distort)

        E, inliers = cv2.findEssentialMat(pts1_undist, pts2_undist, mtx)
        inliers = inliers.ravel()
        pts1_undist = [p for f, p in zip(inliers, pts1_undist) if f >= 1]
        pts2_undist = [p for f, p in zip(inliers, pts2_undist) if f >= 1]

        # print(inliers)
        ret, R, t = recoverPose(E, pts1_undist, pts2_undist, mtx)
        if not ret:
            print("failed", (cam_id1, cam_id2))

        Rt = make_Rt(R, t)
        identity = np.eye(4)[:3]
        proj1 = np.matmul(mtx, identity)
        proj2 = np.matmul(mtx, Rt)

        points3d = [np.zeros(3, float)]
        colors = [cam_id_colors[cam_id1]]
        points3d.append(make_view_position(R, t.ravel()))
        colors.append(cam_id_colors[cam_id2])
        for p1, p2 in zip(pts1_undist, pts2_undist):
            pt3d = geometry.triangulate_nviews(
                np.array([p1, p2]), np.array([proj1, proj2])
            )
            points3d.append(pt3d)
            colors.append((0.5, 0.5, 0.5))

        print(cam_id1, cam_id2)

        # visualize
        
        R1, R2, t_ = cv2.decomposeEssentialMat(E)
        ax = new_plt(world_size=1.5)
        draw_camera(ax, np.eye(3), np.zeros(3, float), cam_id1)
        for _R, _t in [[R1, t_], [R2, t_], [R1, -t_], [R2, -t]]:
            draw_camera(ax, _R, _t.ravel(), cam_id2)
        draw_points3d(
            ax, points3d, colors, world_size=1.5, center=(0, 0, 0), block=True, s=64
        )
       # show_plt(True)

        rvec, _ = cv2.Rodrigues(R)
        tvec = t.ravel()

        for (cam_id1, cam_id2), (pts1, pts2) in corr.items():
            pts1 = np.reshape(pts1, (-1, 2))
            pts2 = np.reshape(pts2, (-1, 2))
            img = np.zeros((h, w * 2, 3), np.uint8)
            for p1, p2 in zip(pts1, pts2):
                x1, y1 = p1.astype(int)
                x2, y2 = p2.astype(int)
#                cv2.line(img, (x1, y1), (x2 + w, y2), (20, 20, 255))
                cv2.circle(img, (x1, y1), 2, (255, 0, 0), -1)
                cv2.circle(img, (w + x2, y2), 2, (255, 0, 0), -1)

            points2d, _ = cv2.projectPoints(np.array(points3d), np.zeros_like(rvec), np.zeros_like(tvec), mtx, distort)
            for p in points2d.reshape((-1, 2)):
                x1, y1 = p.astype(int)
                cv2.circle(img, (x1, y1), 6, (0, 255, 0), 1)

            points2d, _ = cv2.projectPoints(np.array(points3d), rvec, tvec, mtx, distort)
            for p in points2d.reshape((-1, 2)):
                x2, y2 = p.astype(int)
                cv2.circle(img, (w + x2, y2), 6, (0, 255, 0), 1)

            cv2.imshow("", img)
            cv2.waitKey(0)

        Rt_dict[(cam_id1, cam_id2)] = R, t

    print(Rt_dict)
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
                global_R = R  # M[:3, :3]
                global_t = t.ravel()  # M[:3, 3]

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