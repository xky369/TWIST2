"""Analytical FK + joint limits for the G1 dual arm (torso_link frame).

Verbatim copy of the relevant portions of
``rl_ik_solver/.../g1_rl_controller_py/trajectory.py`` so the TWIST2
replay driver reports *actual* EE poses with bit-for-bit the same
formula (and the same pre-FK joint clip) as the rl_ik_solver
baseline. Keeping a local copy avoids dragging the unitree_sdk2py
stack through a cross-package import.

Inputs / outputs
----------------
* ``larm_forward(theta7)`` / ``rarm_forward(theta7)`` return the
  wrist_yaw_link pose in the torso_link frame as a 7-vector
  ``[x, y, z, qw, qx, qy, qz]``. The analytical chain starts at the
  fixed torso_link->shoulder_anchor transform, so the returned pose
  is independent of the 3 waist joints (which is correct on the G1
  because both arms branch off torso_link).
* ``LEFT_JOINT_LIMITS_RAD`` / ``RIGHT_JOINT_LIMITS_RAD`` are the hard
  joint limits used to clip the measured joint readings BEFORE FK.
  rl_ik_solver applies this clip so that motor readings that slip
  1e-4 rad past the URDF limit (sim or real) don't poison the
  metric; we mirror it here for parity.
"""

from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------
# G1 dual-arm joint limits (rad), ordered as
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#    wrist_roll, wrist_pitch, wrist_yaw]
# Identical to G1_JOINT_LIMITS[:7] / [7:] in rl_ik_solver trajectory.py.
# ----------------------------------------------------------------------

_G1_JOINT_LIMITS = [
    (-3.0892, 2.6704),            # L_SHOULDER_PITCH
    (-1.5882, 2.2515),            # L_SHOULDER_ROLL
    (-2.6180, 2.6180),            # L_SHOULDER_YAW
    (-1.0472, 2.0944),            # L_ELBOW
    (-1.972222054, 1.972222054),  # L_WRIST_ROLL
    (-1.614429558, 1.614429558),  # L_WRIST_PITCH
    (-1.614429558, 1.614429558),  # L_WRIST_YAW
    (-3.0892, 2.6704),            # R_SHOULDER_PITCH
    (-2.2515, 1.5882),            # R_SHOULDER_ROLL
    (-2.6180, 2.6180),            # R_SHOULDER_YAW
    (-1.0472, 2.0944),            # R_ELBOW
    (-1.972222054, 1.972222054),  # R_WRIST_ROLL
    (-1.614429558, 1.614429558),  # R_WRIST_PITCH
    (-1.614429558, 1.614429558),  # R_WRIST_YAW
]

LEFT_JOINT_LIMITS_RAD = np.array(_G1_JOINT_LIMITS[:7], dtype=np.float32)
RIGHT_JOINT_LIMITS_RAD = np.array(_G1_JOINT_LIMITS[7:], dtype=np.float32)


# ----------------------------------------------------------------------
# Small homogeneous-transform helpers (identical to rl_ik_solver).
# ----------------------------------------------------------------------


def _t_trans(offset_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = offset_xyz
    return transform


def _t_rot_x(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _t_rot_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = cy * cp
    transform[0, 1] = cy * sp * sr - sy * cr
    transform[0, 2] = cy * sp * cr + sy * sr
    transform[1, 0] = sy * cp
    transform[1, 1] = sy * sp * sr + cy * cr
    transform[1, 2] = sy * sp * cr - cy * sr
    transform[2, 0] = -sp
    transform[2, 1] = cp * sr
    transform[2, 2] = cp * cr
    return transform


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


# ----------------------------------------------------------------------
# Analytical FK (torso_link -> wrist_yaw_link) -- verbatim from
# rl_ik_solver/.../trajectory.py::larm_forward / rarm_forward.
# ----------------------------------------------------------------------


def larm_forward(theta: np.ndarray) -> np.ndarray:
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
    pos = hand[:3, 3]
    quat = _matrix_to_quaternion_wxyz(hand[:3, :3])
    return np.concatenate([pos, quat]).astype(np.float32)


def rarm_forward(theta: np.ndarray) -> np.ndarray:
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
    pos = hand[:3, 3]
    quat = _matrix_to_quaternion_wxyz(hand[:3, :3])
    return np.concatenate([pos, quat]).astype(np.float32)


def clip_left_arm(theta7: np.ndarray) -> np.ndarray:
    return np.clip(theta7, LEFT_JOINT_LIMITS_RAD[:, 0], LEFT_JOINT_LIMITS_RAD[:, 1])


def clip_right_arm(theta7: np.ndarray) -> np.ndarray:
    return np.clip(theta7, RIGHT_JOINT_LIMITS_RAD[:, 0], RIGHT_JOINT_LIMITS_RAD[:, 1])
