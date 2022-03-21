import argparse
from email.mime import image
import numpy as np
import cv2
import glob
from pathlib import Path
import pandas as pd

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


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    all_ids = calib.result.all_ids
    all_cam_ids = sorted(set([c for _, c in all_ids]))

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
    for (cam_id1, cam_id2), (pts1, pts2) in corr.items():
        pts1 = np.reshape(pts1, (-1, 1, 2))
        pts2 = np.reshape(pts2, (-1, 1, 2))
        pts1_undist = geometry.undistort(np.array(pts1), mtx, distort)
        pts2_undist = geometry.undistort(np.array(pts2), mtx, distort)

        E, inliers = cv2.findEssentialMat(pts1_undist, pts2_undist, mtx)
        #print(inliers)
        ret, R, t = recoverPose(E, pts1_undist, pts2_undist, mtx)
        if not ret:
            print("failed", (cam_id1, cam_id2))

        print("OK", (cam_id1, cam_id2), R, t)


if __name__ == "__main__":
    args = parse_args()
    main(args)
