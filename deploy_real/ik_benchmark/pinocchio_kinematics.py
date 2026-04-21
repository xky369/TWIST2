"""Shared Pinocchio-backed kinematic core for the G1 dual-arm IK solvers.

This module is the "library" layer used by both
:class:`PinocchioDualArmDLSIKSolver` (in ``pinocchio_dls_solver.py``)
and :class:`PinocchioDualArmSLSQPIKSolver` (in
``pinocchio_slsqp_solver.py``). Factoring the kinematics out into a
standalone helper class keeps the two solver files focused on the
algorithm they actually implement (DLS update vs. SLSQP outer loop
with analytical gradient), and avoids having to duplicate the
fiddly parts of URDF loading, frame bookkeeping, and safe pinocchio
import.

Public API:
    * :func:`_import_pinocchio_safely` - import pinocchio even when a
      foreign-python-version ROS pinocchio is shadowing the conda
      install via ``source /opt/ros/<distro>/setup.bash``.
    * :data:`DUAL_JOINT_LIMITS` - 14x2 (lower, upper) matching the
      analytical FK in :mod:`g1_kinematics`.
    * :class:`DualIKSolveResult` - result dataclass shared by every
      dual-arm IK solver in this package.
    * :class:`G1DualArmPinocchioKinematics` - builds the Pinocchio
      model from a URDF, resolves the end-effector frames and the 14
      arm joints, caches the constant reference_frame <- root
      transform, and exposes a single ``fk_and_jacobians`` entry
      point that returns the positions, rotations and 6x7 analytical
      Jacobians needed by both DLS and SLSQP.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .g1_kinematics import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS, Pose
except ImportError:
    from g1_kinematics import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS, Pose


# ----------------------------------------------------------------------
# Safe pinocchio import (handles ROS PYTHONPATH shadowing)
# ----------------------------------------------------------------------


_ROS_PY_PATH_RE = re.compile(r"/opt/ros/[^/]+/(?:local/)?lib/python(\d+)\.(\d+)/")


def _import_pinocchio_safely():
    """Import pinocchio, scrubbing stale ROS site-packages only if needed.

    Context: ``source /opt/ros/humble/setup.bash`` prepends
    ``/opt/ros/humble/lib/python3.10/site-packages`` (Python 3.10) to
    both ``PYTHONPATH`` and ``sys.path``. On this workstation the
    conda env runs Python 3.11 with pinocchio installed via cmeel.
    When Python 3.11 tries to import that 3.10 pinocchio it fails
    with ``No module named 'pinocchio.pinocchio_pywrap_default'``
    because the ``.so`` is ABI-incompatible.

    Strategy (in order):

    1. First try a plain ``import pinocchio``. If that works and
       exposes the API we need, return immediately. This is the
       hot path on real robots with one clean install, and also
       the hot path when this helper is called a second time in
       the same process (Python C extensions cannot be unloaded,
       so we must reuse an already-imported module instead of
       trying to re-initialize it).

    2. Only if step 1 fails do we hide any ``sys.path`` / ``PYTHONPATH``
       entries that refer to a ROS Python ``X.Y`` that does not
       match the current interpreter, drop any half-imported
       ``pinocchio`` from ``sys.modules``, retry the import, and
       then restore the paths so that unrelated downstream code
       that depends on them keeps working.
    """
    try:
        import pinocchio as pin  # type: ignore

        pin.buildModelFromUrdf  # noqa: B018
        pin.computeJointJacobians  # noqa: B018
        pin.log3  # noqa: B018
        return pin
    except (ImportError, AttributeError):
        pass

    cur_major, cur_minor = sys.version_info.major, sys.version_info.minor

    def _is_foreign_ros_path(p: str) -> bool:
        m = _ROS_PY_PATH_RE.search(p)
        if m is None:
            return False
        return (int(m.group(1)), int(m.group(2))) != (cur_major, cur_minor)

    removed_sys_path: list[tuple[int, str]] = []
    for idx, p in enumerate(list(sys.path)):
        if _is_foreign_ros_path(p):
            removed_sys_path.append((idx, p))
    for _, p in removed_sys_path:
        while p in sys.path:
            sys.path.remove(p)

    original_pythonpath = os.environ.get("PYTHONPATH")
    if original_pythonpath is not None:
        kept = [
            p for p in original_pythonpath.split(os.pathsep)
            if p and not _is_foreign_ros_path(p)
        ]
        if kept:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            del os.environ["PYTHONPATH"]

    for mod_name in list(sys.modules):
        if mod_name == "pinocchio" or mod_name.startswith("pinocchio."):
            del sys.modules[mod_name]

    try:
        import pinocchio as pin  # type: ignore
    finally:
        for idx, p in removed_sys_path:
            if p not in sys.path:
                sys.path.insert(min(idx, len(sys.path)), p)
        if original_pythonpath is not None:
            os.environ["PYTHONPATH"] = original_pythonpath

    return pin


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


def _resolve_default_urdf() -> Path:
    """Locate the vendored G1 locked-waist URDF.

    Search order:

    1. ``$TWIST2_G1_LOCKED_WAIST_URDF`` (explicit override).
    2. ``<TWIST2>/assets/g1/g1_29dof_simple_collision_lock_waist_with_hand.urdf``
       -- the file we ship alongside this vendored module.
    3. The original absolute path at the author's workstation, as a
       last-ditch fallback so existing setups keep working.

    The first path that exists on disk wins. If none exists we still
    return option (2) so callers see a descriptive FileNotFoundError
    pointing at the expected TWIST2 asset location.
    """

    env_override = os.environ.get("TWIST2_G1_LOCKED_WAIST_URDF")
    if env_override:
        p = Path(env_override).expanduser().resolve()
        if p.is_file():
            return p

    # ``<TWIST2>/deploy_real/ik_benchmark/pinocchio_kinematics.py``
    # -> parents[2] == ``<TWIST2>``.
    twist2_root = Path(__file__).resolve().parents[2]
    vendored = twist2_root / "assets" / "g1" / "g1_29dof_simple_collision_lock_waist_with_hand.urdf"
    if vendored.is_file():
        return vendored

    legacy = Path(
        "/home/rail/rail-unitree/rail_unitree_g1/source/assets/"
        "g1_simple_collision_lock_waist_with_hand/"
        "g1_29dof_simple_collision_lock_waist_with_hand.urdf"
    )
    if legacy.is_file():
        return legacy

    return vendored


DEFAULT_URDF_PATH = _resolve_default_urdf()

DEFAULT_LEFT_EE_FRAME = "left_wrist_yaw_link"
DEFAULT_RIGHT_EE_FRAME = "right_wrist_yaw_link"

# Targets in this codebase are authored in the ``torso_link`` frame
# (that is what ``g1_kinematics.left_arm_fk`` returns, because the
# analytical FK chain starts at the shoulder anchor which is rigidly
# attached to ``torso_link`` on the lock-waist G1). Pinocchio builds
# the URDF with ``pelvis`` as the root, so we internally transform
# every target from the reference frame to pelvis before solving.
DEFAULT_REFERENCE_FRAME = "torso_link"


DEFAULT_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

DEFAULT_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


DUAL_JOINT_LIMITS = np.vstack([LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS])


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------


@dataclass
class DualIKSolveResult:
    """Return type for every dual-arm IK solver in this package.

    Attributes
    ----------
    q:
        14-vector of solved joint angles ``[left7, right7]``, already
        clipped to the solver's hard joint limits.
    success:
        ``True`` iff BOTH arms satisfied ``||p_err||_2 < tol_pos`` and
        ``||omega_err||_2 < tol_rot`` at the returned ``q``.
    iterations:
        Number of outer-loop iterations the solver used (definition
        depends on the backend: for DLS it is literal gradient steps;
        for SLSQP it is ``scipy.optimize.OptimizeResult.nit``).
    left_pos_error_m, right_pos_error_m:
        Per-arm Euclidean position residual at the returned ``q``.
    left_rot_error_deg, right_rot_error_deg:
        Per-arm rotation-vector residual at the returned ``q``,
        reported in degrees so that it reads naturally in paper
        tables (the internal metric is the norm of ``log3(R_err)``).
    pos_error_m, rot_error_deg:
        Aggregates: the mean of the left and right per-arm residuals
        above. Handy for at-a-glance one-line logging, but note that
        you should still report the per-arm numbers in publications.
    """

    q: np.ndarray
    success: bool
    iterations: int
    left_pos_error_m: float
    right_pos_error_m: float
    left_rot_error_deg: float
    right_rot_error_deg: float
    pos_error_m: float
    rot_error_deg: float


# ----------------------------------------------------------------------
# Quaternion helpers (closed form, no scipy)
# ----------------------------------------------------------------------


def wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Closed-form (w, x, y, z) quaternion -> 3x3 rotation.

    Dependency-free to avoid ``scipy.spatial.transform.Rotation``
    construction on every IK call (~100 us per call).
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Closed-form 3x3 rotation -> (w, x, y, z) unit quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return q / max(np.linalg.norm(q), 1e-12)


# ----------------------------------------------------------------------
# Shared kinematic core
# ----------------------------------------------------------------------


class G1DualArmPinocchioKinematics:
    """URDF-backed FK + analytical Jacobians for the G1 two-arm chain.

    This class does not know anything about IK. It is solely
    responsible for:

        * loading the URDF with Pinocchio (handling ROS PYTHONPATH
          shadowing via :func:`_import_pinocchio_safely`),
        * resolving the end-effector and reference frames,
        * resolving the 14 arm joint names to Pinocchio ``idx_q``,
        * caching the constant reference_frame <- root transform
          (exact for lock-waist G1 because ``torso_link`` is
          attached to ``pelvis`` via fixed joints),
        * evaluating FK and the 6x7 analytical Jacobian for each
          arm in one batched call that reuses ``data``.

    Both DLS and SLSQP solvers own an instance and call
    :meth:`fk_and_jacobians` once per outer iteration.

    Parameters are identical to the ones documented on the two
    solver classes; see :class:`PinocchioDualArmDLSIKSolver` or
    :class:`PinocchioDualArmSLSQPIKSolver`.
    """

    def __init__(
        self,
        urdf_path: Optional[Path | str] = None,
        *,
        left_ee_frame: str = DEFAULT_LEFT_EE_FRAME,
        right_ee_frame: str = DEFAULT_RIGHT_EE_FRAME,
        reference_frame: str = DEFAULT_REFERENCE_FRAME,
        left_arm_joints: Sequence[str] = DEFAULT_LEFT_ARM_JOINTS,
        right_arm_joints: Sequence[str] = DEFAULT_RIGHT_ARM_JOINTS,
    ) -> None:
        self.pin = _import_pinocchio_safely()
        pin = self.pin

        urdf = Path(urdf_path).expanduser().resolve() if urdf_path is not None else DEFAULT_URDF_PATH
        if not urdf.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf}")

        self.urdf_path = urdf
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)

        if not self.model.existFrame(left_ee_frame):
            raise ValueError(f"URDF has no frame named '{left_ee_frame}'")
        if not self.model.existFrame(right_ee_frame):
            raise ValueError(f"URDF has no frame named '{right_ee_frame}'")
        self.left_ee_id = int(self.model.getFrameId(left_ee_frame))
        self.right_ee_id = int(self.model.getFrameId(right_ee_frame))

        self.left_q_idx = self._resolve_joint_indices(left_arm_joints)
        self.right_q_idx = self._resolve_joint_indices(right_arm_joints)
        assert self.left_q_idx.size == 7 and self.right_q_idx.size == 7

        # Non-arm slots are frozen at the neutral configuration: they
        # do not live on the arm kinematic chain on the lock-waist
        # G1, so their value cannot affect FK on the end-effectors.
        self._q_full = pin.neutral(self.model).astype(np.float64)

        # Precompute the constant reference_frame <- pelvis transform.
        # Valid because the lock-waist URDF attaches ``torso_link`` to
        # ``pelvis`` via fixed joints.
        if (
            reference_frame == "pelvis"
            or reference_frame == ""
            or reference_frame == self.model.frames[0].name
        ):
            self._ref_is_root = True
            self._R_root_ref = np.eye(3, dtype=np.float64)
            self._p_root_ref = np.zeros(3, dtype=np.float64)
            self.reference_frame = "pelvis"
        else:
            if not self.model.existFrame(reference_frame):
                raise ValueError(f"URDF has no frame named '{reference_frame}'")
            pin.forwardKinematics(self.model, self.data, self._q_full)
            pin.updateFramePlacements(self.model, self.data)
            oMref = self.data.oMf[int(self.model.getFrameId(reference_frame))]
            self._ref_is_root = False
            self._R_root_ref = np.asarray(oMref.rotation, dtype=np.float64).copy()
            self._p_root_ref = np.asarray(oMref.translation, dtype=np.float64).copy()
            self.reference_frame = reference_frame

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    def _resolve_joint_indices(self, names: Sequence[str]) -> np.ndarray:
        idx = np.zeros(len(names), dtype=np.int64)
        for k, name in enumerate(names):
            if not self.model.existJointName(name):
                raise ValueError(f"URDF has no joint named '{name}'")
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Joint '{name}' resolved to universe (id=0); URDF mismatch")
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Joint '{name}' has nq={joint.nq}, nv={joint.nv}; dual-arm IK assumes 1-DoF revolute"
                )
            idx[k] = int(joint.idx_q)
        return idx

    # ------------------------------------------------------------------
    # Forward kinematics + Jacobian (analytical, world-aligned)
    # ------------------------------------------------------------------

    def fk_and_jacobians(
        self, q14: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched FK + Jacobians for both wrists.

        Returns
        -------
        p_L, R_L, p_R, R_R : end-effector position (m) and rotation
            (3x3) in the pelvis (URDF root) frame.
        J_L_arm, J_R_arm   : 6x7 analytical Jacobians in
            ``LOCAL_WORLD_ALIGNED`` convention (origin at the wrist,
            axes aligned with the world / pelvis frame). The first
            three rows are the translational part and map q_dot to
            linear velocity of the wrist origin; the last three rows
            are the angular part and map q_dot to angular velocity
            in the world frame. Only the columns corresponding to
            the seven arm joints are returned; all others are zero
            on the lock-waist G1 anyway because the arms and legs
            are independent kinematic sub-trees.
        """
        pin = self.pin

        q = self._q_full
        q[self.left_q_idx] = q14[:7]
        q[self.right_q_idx] = q14[7:]

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        oMl = self.data.oMf[self.left_ee_id]
        oMr = self.data.oMf[self.right_ee_id]

        J_L = pin.getFrameJacobian(self.model, self.data, self.left_ee_id, pin.LOCAL_WORLD_ALIGNED)
        J_R = pin.getFrameJacobian(self.model, self.data, self.right_ee_id, pin.LOCAL_WORLD_ALIGNED)

        J_L_arm = J_L[:, self.left_q_idx]
        J_R_arm = J_R[:, self.right_q_idx]

        return (
            np.asarray(oMl.translation, dtype=np.float64).copy(),
            np.asarray(oMl.rotation, dtype=np.float64).copy(),
            np.asarray(oMr.translation, dtype=np.float64).copy(),
            np.asarray(oMr.rotation, dtype=np.float64).copy(),
            np.asarray(J_L_arm, dtype=np.float64),
            np.asarray(J_R_arm, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Reference-frame conversions
    # ------------------------------------------------------------------

    def ref_to_root(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """Lift a ``(pos, quat_wxyz)`` pose from ``reference_frame`` to pelvis."""
        R_tgt_ref = wxyz_to_rotmat(np.asarray(pose.quat_wxyz, dtype=np.float64))
        p_tgt_ref = np.asarray(pose.pos, dtype=np.float64)
        if self._ref_is_root:
            return p_tgt_ref, R_tgt_ref
        R_tgt_root = self._R_root_ref @ R_tgt_ref
        p_tgt_root = self._R_root_ref @ p_tgt_ref + self._p_root_ref
        return p_tgt_root, R_tgt_root

    def root_to_ref(
        self, p_root: np.ndarray, R_root: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a pelvis-frame pose back into ``reference_frame``."""
        if self._ref_is_root:
            return p_root, R_root
        R_ref = self._R_root_ref.T @ R_root
        p_ref = self._R_root_ref.T @ (p_root - self._p_root_ref)
        return p_ref, R_ref

    # ------------------------------------------------------------------
    # Convenience FK (mirrors g1_kinematics.left_arm_fk / right_arm_fk)
    # ------------------------------------------------------------------

    def forward_kinematics(self, q14: np.ndarray) -> tuple[Pose, Pose]:
        p_L, R_L, p_R, R_R, _, _ = self.fk_and_jacobians(np.asarray(q14, dtype=np.float64))
        p_L_ref, R_L_ref = self.root_to_ref(p_L, R_L)
        p_R_ref, R_R_ref = self.root_to_ref(p_R, R_R)
        return (
            Pose(pos=p_L_ref, quat_wxyz=rotmat_to_wxyz(R_L_ref)),
            Pose(pos=p_R_ref, quat_wxyz=rotmat_to_wxyz(R_R_ref)),
        )
