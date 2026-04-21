from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


LEFT_JOINT_LIMITS = np.array(
    [
        [-3.0892, 2.6704],      # shoulder pitch
        [-1.5882, 2.2515],      # shoulder roll
        [-2.6180, 2.6180],      # shoulder yaw
        [-1.0472, 2.0944],      # elbow
        [-1.972222054, 1.972222054],  # wrist roll
        [-1.614429558, 1.614429558],  # wrist pitch
        [-1.614429558, 1.614429558],  # wrist yaw
    ],
    dtype=np.float64,
)

RIGHT_JOINT_LIMITS = np.array(
    [
        [-3.0892, 2.6704],      # shoulder pitch
        [-2.2515, 1.5882],      # shoulder roll
        [-2.6180, 2.6180],      # shoulder yaw
        [-1.0472, 2.0944],      # elbow
        [-1.972222054, 1.972222054],  # wrist roll
        [-1.614429558, 1.614429558],  # wrist pitch
        [-1.614429558, 1.614429558],  # wrist yaw
    ],
    dtype=np.float64,
)


@dataclass
class Pose:
    pos: np.ndarray     # shape (3,)
    quat_wxyz: np.ndarray  # shape (4,)


def _t_trans(v: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = v
    return t


def _t_rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    t = np.eye(4, dtype=np.float64)
    t[0, 0] = cy * cp
    t[0, 1] = cy * sp * sr - sy * cr
    t[0, 2] = cy * sp * cr + sy * sr
    t[1, 0] = sy * cp
    t[1, 1] = sy * sp * sr + cy * cr
    t[1, 2] = sy * sp * cr - cy * sr
    t[2, 0] = -sp
    t[2, 1] = cp * sr
    t[2, 2] = cp * cr
    return t


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def pose_error_6d(target: Pose, current: Pose) -> np.ndarray:
    pos_err = target.pos - current.pos
    rt = Rotation.from_quat(wxyz_to_xyzw(target.quat_wxyz))
    rc = Rotation.from_quat(wxyz_to_xyzw(current.quat_wxyz))
    ori_err = (rt * rc.inv()).as_rotvec()
    return np.concatenate([pos_err, ori_err], axis=0)


def left_arm_fk(theta: np.ndarray) -> Pose:
    shoulder = (
        _t_trans(np.array([0.0039563, 0.10022, 0.24778], dtype=np.float64))
        @ _t_rot_rpy(np.array([0.27931, 0.000054949, -0.00019159], dtype=np.float64))
        @ _t_rot_y(float(theta[0]))
        @ _t_trans(np.array([0.0, 0.038, -0.013831], dtype=np.float64))
        @ _t_rot_rpy(np.array([-0.27925, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_x(float(theta[1]))
        @ _t_trans(np.array([0.0, 0.00624, -0.1032], dtype=np.float64))
        @ _t_rot_z(float(theta[2]))
    )
    elbow = _t_trans(np.array([0.015783, 0.0, -0.080518], dtype=np.float64)) @ _t_rot_y(float(theta[3]))
    wrist = (
        _t_trans(np.array([0.1, 0.00188791, -0.01], dtype=np.float64))
        @ _t_rot_x(float(theta[4]))
        @ _t_trans(np.array([0.038, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_y(float(theta[5]))
        @ _t_trans(np.array([0.046, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_z(float(theta[6]))
    )
    hand = shoulder @ elbow @ wrist
    quat = xyzw_to_wxyz(Rotation.from_matrix(hand[:3, :3]).as_quat())
    return Pose(pos=hand[:3, 3].copy(), quat_wxyz=quat)


def right_arm_fk(theta: np.ndarray) -> Pose:
    shoulder = (
        _t_trans(np.array([0.0039563, -0.10021, 0.24778], dtype=np.float64))
        @ _t_rot_rpy(np.array([-0.27931, 0.000054949, 0.00019159], dtype=np.float64))
        @ _t_rot_y(float(theta[0]))
        @ _t_trans(np.array([0.0, -0.038, -0.013831], dtype=np.float64))
        @ _t_rot_rpy(np.array([0.27925, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_x(float(theta[1]))
        @ _t_trans(np.array([0.0, -0.00624, -0.1032], dtype=np.float64))
        @ _t_rot_z(float(theta[2]))
    )
    elbow = _t_trans(np.array([0.015783, 0.0, -0.080518], dtype=np.float64)) @ _t_rot_y(float(theta[3]))
    wrist = (
        _t_trans(np.array([0.1, -0.00188791, -0.01], dtype=np.float64))
        @ _t_rot_x(float(theta[4]))
        @ _t_trans(np.array([0.038, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_y(float(theta[5]))
        @ _t_trans(np.array([0.046, 0.0, 0.0], dtype=np.float64))
        @ _t_rot_z(float(theta[6]))
    )
    hand = shoulder @ elbow @ wrist
    quat = xyzw_to_wxyz(Rotation.from_matrix(hand[:3, :3]).as_quat())
    return Pose(pos=hand[:3, 3].copy(), quat_wxyz=quat)
