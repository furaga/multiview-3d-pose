import argparse
import cv2
import time
import os
import pickle
import random
from glob import glob
import numpy as np
from typing import Dict, Set, Any, List, NamedTuple, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


def make_3d_points(n):
    objectPoints = []
    p = (0.0, 0.0, 0.0)
    objectPoints.append(p)
    for i in range(n):
        p = [2 * random.random() - 1 for _ in range(3)]
        objectPoints.append(p)
    objectPoints = np.array(objectPoints)
    return objectPoints


def Rx(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.reshape(np.array([1, 0, 0, 0, cos, -sin, 0, sin, cos]), (3, 3))


def Ry(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.reshape(np.array([cos, 0, sin, 0, 1, 0, -sin, 0, cos]), (3, 3))


def Rz(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.reshape(np.array([cos, -sin, 0, sin, cos, 0, 0, 0, 1]), (3, 3))


def make_view_poses(n, distance):
    rvecs = []
    tvecs = []

    for i in range(n):
        angle = 2 * np.pi * i / n
        R = np.matmul(Ry(angle), Rx(0.1))
        rvec, _ = cv2.Rodrigues(R)
        rvecs.append(rvec)
        tvecs.append(np.array([0, 0, distance]))

    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return rvecs, tvecs
