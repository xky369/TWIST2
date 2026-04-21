"""Vendored IK benchmark utilities from rl_ik_solver.

Copied verbatim from ``rl_ik_solver/trajectory/ik_benchmark`` so the
TWIST2 trajectory-replay driver (``trajectory_replay_sim2sim.py``) has
no runtime dependency on an external sibling repo. This keeps the
server upload self-contained.

Update policy: if the rl_ik_solver side ever diverges, re-copy these
three files AND ``g1_29dof_simple_collision_lock_waist_with_hand.urdf``
(see ``assets/g1/``) together so kinematics stay consistent.
"""

from .g1_kinematics import Pose
from .pinocchio_kinematics import (
    DEFAULT_LEFT_ARM_JOINTS,
    DEFAULT_LEFT_EE_FRAME,
    DEFAULT_REFERENCE_FRAME,
    DEFAULT_RIGHT_ARM_JOINTS,
    DEFAULT_RIGHT_EE_FRAME,
    DEFAULT_URDF_PATH,
    DUAL_JOINT_LIMITS,
    DualIKSolveResult,
    G1DualArmPinocchioKinematics,
)
from .pinocchio_slsqp_solver import PinocchioDualArmSLSQPIKSolver

__all__ = [
    "Pose",
    "DEFAULT_LEFT_ARM_JOINTS",
    "DEFAULT_LEFT_EE_FRAME",
    "DEFAULT_REFERENCE_FRAME",
    "DEFAULT_RIGHT_ARM_JOINTS",
    "DEFAULT_RIGHT_EE_FRAME",
    "DEFAULT_URDF_PATH",
    "DUAL_JOINT_LIMITS",
    "DualIKSolveResult",
    "G1DualArmPinocchioKinematics",
    "PinocchioDualArmSLSQPIKSolver",
]
