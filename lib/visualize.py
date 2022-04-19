import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from . import geometry


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
    return mean


def plot_camera(ax, R, t, id, size=1):
    x = [size, 0, 0]
    y = [0, size, 0]
    z = [0, 0, size]
    c = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(3):
        p0 = geometry.make_view_position(R, t)
        p = np.array([x[i], y[i], z[i]])
        p = np.matmul(R.T, p.T).T + p0
        ax.plot(
            [p0[0], p[0]], [p0[1], p[1]], [p0[2], p[2]], "o-", c=c[i], ms=4, mew=0.5
        )
        ax.text(p0[0], p0[1], p0[2], id, None)


def plot_ray(ax, p0, dir, color=(0, 0, 0), length=3):
    p = p0 + dir * length
    ax.plot([p0[0], p[0]], [p0[1], p[1]], [p0[2], p[2]], "o-", c=color, ms=4, mew=0.5)
