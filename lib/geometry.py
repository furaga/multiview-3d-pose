import cv2
import numpy as np
from typing import Any, List, NamedTuple, Optional, Tuple, Dict


def calc_view_position(pose):
    R, t = pose[:3, :3], pose[:3, 3]
    p = -np.matmul(np.linalg.inv(R), t.T).T
    return p


def normalize(vec):
    return vec / np.linalg.norm(vec)


def triangulate_nviews(
    points2d: np.ndarray, proj_mats: np.ndarray, show_error=False
) -> np.ndarray:
    assert points2d.shape[1] == 2
    nviews = points2d.shape[0]
    assert nviews == len(proj_mats)

    A = np.zeros((3 * nviews, 4 + nviews))
    for i in range(nviews):
        yoffset = 3 * i
        A[yoffset : yoffset + 3, :4] = -proj_mats[i]
        A[yoffset + 0, 4 + i] = points2d[i][0]
        A[yoffset + 1, 4 + i] = points2d[i][1]
        A[yoffset + 2, 4 + i] = 1

    # SVDを使って劣決定連立方程式 Ax=0 を解く
    _, _, VT = cv2.SVDecomp(A)
    x_alphas = VT[-1]
    x = x_alphas[:3] / x_alphas[3]

    return x


def calc_trianglate_error(points2d, proj_mats, x):
    errs = []
    for pt2d, proj in zip(points2d, proj_mats):
        v = np.array([*x, 1], float).T
        reproj = np.matmul(proj, v).ravel()
        reproj = reproj[:2] / reproj[2]
        err = np.linalg.norm(pt2d - reproj)
        errs.append(err)
    return errs


def triangulate_nviews_by2(points2d: np.ndarray, proj_mats: np.ndarray) -> np.ndarray:
    n = len(points2d)

    points3d = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            p = triangulate_nviews(
                np.take(points2d, [i, j], axis=0),
                np.take(proj_mats, [i, j], axis=0),
                True,
            )
            errs = calc_trianglate_error(points2d, proj_mats, p)
            points3d.append(p)

    if len(points3d) <= 0:
        return False, None

    x = np.mean(points3d, axis=0)

    errs = []
    for pt2d, proj in zip(points2d, proj_mats):
        v = np.array([*x, 1], float).T
        reproj = np.matmul(proj, v).ravel()
        reproj = reproj[:2] / reproj[2]
        err = np.linalg.norm(pt2d - reproj)
        errs.append(err)

    return True, x


def calc_depth(pose, X):
    R = pose[:3, :3]
    t = pose[:3, 3]
    return np.matmul(R, X.T).T[2] + t[2]


def undistort(points, mtx, distort):
    if np.all(distort == 0):
        return points
    points = np.array(points)
    points = cv2.undistortPoints(points, mtx, distort)
    points = np.reshape(points, (-1, 2))
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    points[:, 0] = points[:, 0] * fx + cx
    points[:, 1] = points[:, 1] * fy + cy
    return points


def calc_view_angles(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    v1 = np.matmul(R1, np.array([0, 0, 1]))
    v2 = np.matmul(R2, np.array([0, 0, 1]))
    dot = np.dot(v1, v2)
    degree = np.degrees(np.arccos(dot))
    return degree


def calc_ray_angles(ray1, ray2):
    ray1 /= np.linalg.norm(ray1) + 1e-8
    ray2 /= np.linalg.norm(ray2) + 1e-8
    dot = np.dot(ray1.ravel(), ray2.ravel())
    degree = np.degrees(np.arccos(dot))
    return degree


def calc_rays(points, cam_pose, mtx, distort):
    points = undistort(points, mtx, distort)
    cam_pose = np.array(cam_pose)

    f = np.array([[mtx[0, 0], mtx[1, 1]]])
    c = np.array([[mtx[0, 2], mtx[1, 2]]])

    n = len(points)
    uv = (points - c) / f
    p = np.concatenate([uv, np.ones((n, 1), float)], axis=1)

    R, t = cam_pose[:, :3], cam_pose[:, 3]
    X = np.matmul(np.linalg.inv(R), (p - t).T).T

    # # ちゃんともとに戻るかチェック
    # imgpts, _ = cv2.projectPoints(np.array(X), R, t, mtx, 0)
    # err = np.linalg.norm(imgpts.ravel() - points.ravel())
    # assert err < 1, f"err={err}"

    p0 = calc_view_position(cam_pose)
    dirs = X - p0
    length = np.linalg.norm(dirs, axis=1)
    dirs /= np.expand_dims(length, axis=1)

    # # ちゃんともとに戻るかチェック
    # X2 = p0 + dirs
    # imgpts, _ = cv2.projectPoints(np.array(X2), R, t, mtx, 0)
    # err = np.linalg.norm(imgpts.ravel() - points.ravel())
    # assert err < 1, f"err={err}"

    return p0, dirs


def line_distance(
    p0: np.ndarray, dir: np.ndarray, pt3d: np.ndarray, axis=None
) -> float:
    c = np.cross(dir, pt3d - p0, axis=axis)
    return np.linalg.norm(c, axis=axis)


def closest_point_to_line(
    p0: np.ndarray, dir: np.ndarray, pt3d: np.ndarray, axis=None
) -> float:
    return p0 + dir * np.dot(dir, pt3d - p0)


def intersect_line_triangle(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, p0: np.ndarray, dir: np.ndarray
) -> Tuple[bool, np.ndarray, np.ndarray]:
    v0 = C - A
    v1 = B - A
    n = np.array(
        [
            v0[1] * v1[2] - v0[2] * v1[1],
            v0[2] * v1[0] - v0[0] * v1[2],
            v0[0] * v1[1] - v0[1] * v1[0],
        ]
    )
    k = (np.dot(A, n) - np.dot(p0, n)) / np.dot(dir, n)
    if abs(k) < 1e-8:
        return False, None, None
    P = p0 + dir * k
    v2 = P - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return True, P, np.array((u, v))


def calc_pixel_size(pt3d, pose, mtx, distort, step=1):
    if len(pt3d.shape) == 1:
        R, t = pose[:3, :3], pose[:3, 3]
        kp, _ = cv2.projectPoints(np.array([pt3d]), R, t, mtx, distort)
        x, y = kp.ravel().astype(int)
        nb = (x + step, y)
        _, dirs = calc_rays(np.array([nb], float), pose, mtx, distort)
        nb_dir = dirs[0]
        view_pos = calc_view_position(pose)
        distance = line_distance(view_pos, nb_dir, pt3d)
        return distance
    else:
        R, t = pose[:3, :3], pose[:3, 3]
        kp, _ = cv2.projectPoints(pt3d, R, t, mtx, distort)
        kp = kp.reshape((-1, 2))
        nb = kp + np.array([(step, 0)], float)
        _, nb_dir = calc_rays(nb, pose, mtx, distort)
        view_pos = calc_view_position(pose)
        distance = line_distance(view_pos, nb_dir, pt3d, axis=1)
        return distance


def intersect_line_bounding_box(p0, p1, box_min, box_max):
    def _calc_interp(v1, v2, r):
        if abs(v1 - v2) < 1e-8:
            return -np.inf
        t = (r - v1) / (v2 - v1)
        return t

    def _contains(p, inds):
        for i in inds:
            if p[i] < box_min[i] or box_max[i] < p[i]:
                return False
        return True

    def _calc(ind0, r, ind12):
        t = _calc_interp(p0[ind0], p1[ind0], r)
        if 0 <= t <= 1:
            hit_point = (1 - t) * p0 + t * p1
            if _contains(hit_point, ind12):
                return True, hit_point
        return False, None

    ret, hit_point = _calc(0, box_min[0], [1, 2])
    if ret:
        return hit_point
    ret, hit_point = _calc(0, box_max[0], [1, 2])
    if ret:
        return hit_point

    ret, hit_point = _calc(1, box_min[1], [0, 2])
    if ret:
        return hit_point
    ret, hit_point = _calc(1, box_max[1], [0, 2])
    if ret:
        return hit_point

    ret, hit_point = _calc(2, box_min[2], [0, 1])
    if ret:
        return hit_point
    ret, hit_point = _calc(2, box_max[2], [0, 1])
    if ret:
        return hit_point

    return None


def point_plane_distance(pt, p0, dir):
    return abs(np.dot(pt, dir) - np.dot(p0, dir))


def oriented_point_plane_distance(pt, p0, dir):
    return np.dot(pt, dir) - np.dot(p0, dir)


def line_line_intersect(p1, dir1, p3, dir2):
    p2 = p1 + dir1
    p4 = p3 + dir2

    p13 = p1 - p3
    p43 = p4 - p3
    if np.max(np.abs(p43)) < 1e-8:
        return False, np.zeros(3, float)

    p21 = p2 - p1
    if np.max(np.abs(p21)) < 1e-8:
        return False, np.zeros(3, float)

    d1343 = np.dot(p13, p43)
    d4321 = np.dot(p43, p21)
    d1321 = np.dot(p13, p21)
    d4343 = np.dot(p43, p43)
    d2121 = np.dot(p21, p21)

    denom = d2121 * d4343 - d4321 * d4321

    if abs(denom) < 1e-8:
        return False, np.zeros(3, float)

    numer = d1343 * d4321 - d1321 * d4343

    muab = [
        numer / denom,
        (d1343 + d4321 * numer / denom) / d4343,
    ]

    pa = p1 + muab[0] * p21
    pb = p3 + muab[1] * p43

    pt3d = (pa + pb) / 2
    return True, pt3d


def make_view_position(R, t):
    p = -np.matmul(R.T, t.T).T
    return p
