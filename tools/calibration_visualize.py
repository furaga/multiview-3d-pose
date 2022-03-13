import argparse
from email.mime import image
import numpy as np
import cv2
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random

import calib.result 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def draw_points3d(
    points3d, colors, world_size=0.5, center=None, block=True, s=8
):
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
    assert len(calib.result.rvecs) == len(calib.result.tvecs)
    assert len(calib.result.all_ids) == len(calib.result.tvecs)

    # args.out_dir.mkdir(exist_ok=True)
    
    all_cam_ids = sorted(set([c for _, c in calib.result.all_ids]))
    cam_id_colors = {
        c: (
            random.random(),
            random.random(),
            random.random(),
        ) for c in all_cam_ids
    }
    
    cam_id_colors[all_cam_ids[0]] = (1, 0, 0)
    cam_id_colors[all_cam_ids[1]] = (1, 1, 0)
    cam_id_colors[all_cam_ids[2]] = (1, 0, 1)
    cam_id_colors[all_cam_ids[3]] = (0, 1, 1)
    
    print(cam_id_colors)

    groups = {}
    for rvec, t, (frame, cam_id) in zip(calib.result.rvecs, calib.result.tvecs, calib.result.all_ids):
        groups.setdefault(frame, []).append((cam_id, rvec, t))

    edges = {}        
    for f, ls in groups.items():
        if len(ls) <= 1:
            continue
        ls = sorted(ls)
        for i in range(1, len(ls)):
            cam_id1, rvec1, t1 = ls[0]
            cam_id2, rvec2, t2 = ls[i]
            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            Rt1 = make_Rt(R1, t1, expand=True)
            Rt2 = make_Rt(R2, t2)
            iRt1 = np.linalg.inv(Rt1)
            rel = np.matmul(Rt2, iRt1) # 逆？
            edges.setdefault((cam_id1, cam_id2), []).append(rel)
            
    points3d = [np.zeros(3, float)]
    colors = [cam_id_colors[all_cam_ids[0]]]
    
    cam_positions = [np.zeros(3, float)]
    print(all_cam_ids)
    
    print(list(edges.keys()))
    for i in range(1, len(all_cam_ids)):
        cam_id2 = all_cam_ids[i]
        for j in range(i):
            if j != 0:
                continue # todo
            cam_id1 = all_cam_ids[j]
            es = []
            if (cam_id1, cam_id2) in edges:
                es = edges[(cam_id1, cam_id2)]
            if (cam_id2, cam_id1) in edges:
                es = edges[(cam_id2, cam_id1)]
            for rel in es:
                R = rel[:3, :3]
                t = rel[:3, 3].ravel()
                pt3d = make_view_position(R, t)
                points3d.append(pt3d)
                colors.append(cam_id_colors[cam_id2])
    
    draw_points3d(points3d, colors, world_size=2, center=(0, 0, 0), block=True, s=8)

if __name__ == "__main__":
    args = parse_args()
    main(args)
