"""Pinocchio-backed SLSQP IK for the G1 dual-arm upper body.

Formulation
-----------
We solve, for the 14-DoF concatenated joint vector ``q = [left7; right7]``,

    minimize   f(q) = 1/2 ||W * e_L(q)||^2
                   + 1/2 ||W * e_R(q)||^2
                   + 1/2 * reg_weight * ||q - q_seed||^2

    subject to q_lower <= q <= q_upper

where ``e_i(q) = [p_tgt_i - p_cur_i(q);  log3(R_tgt_i @ R_cur_i(q)^T)]``
is the 6D pose error of the ``i``-th arm in the pelvis (URDF root)
frame, and ``W = diag(pos_weight, pos_weight, pos_weight, rot_weight,
rot_weight, rot_weight)``.

The regularization term ``1/2 * reg_weight * ||q - q_seed||^2``
biases the solution toward the seed ``q_seed`` (typically the
previous step's solution). It is what keeps SLSQP choosing
smooth, close-to-current-pose IK branches at redundant
configurations instead of jumping across the null space, and it
also stabilizes the QP sub-problem.

Analytical gradient
-------------------
The NLP is passed to ``scipy.optimize.minimize(method="SLSQP")`` with
an **analytical gradient** computed from the same Pinocchio
Jacobian used by the DLS solver. Under the world-frame linearization

    d(e_i) / dq = -J_i(q)                                          (6x14)

with ``J_i`` the 6x14 arm Jacobian in ``LOCAL_WORLD_ALIGNED`` form
(nonzero only in the 7 columns of arm ``i``), the gradient of the
residual part of ``f`` is

    grad_f_res(q) = -J_L^T (W^T W) e_L(q)  -  J_R^T (W^T W) e_R(q)  (14,)

and the regularization adds ``reg_weight * (q - q_seed)``. Passing
this exact gradient to SLSQP avoids its default 14-finite-difference
call per objective evaluation, which is where the non-pinocchio
SLSQP implementation used to burn most of its wall-clock time.

The rotational Jacobian is strictly accurate only to first order
(the true sensitivity ``d log3(R_err)/dq`` involves the left
log-Jacobian of ``R_err``), but the dominant term is ``-J_rot`` and
this is what pretty much every standard IK implementation uses. In
practice SLSQP converges in 5-15 iterations for the targets we
care about.

Public API
----------
:class:`PinocchioDualArmSLSQPIKSolver` exposes the same ``solve`` and
``forward_kinematics`` interface as :class:`PinocchioDualArmDLSIKSolver`
(including returning the shared :class:`DualIKSolveResult` dataclass)
so ``ik_controller.py`` and the comparison scripts can swap between
the two solvers by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.optimize import Bounds, minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pinocchio_slsqp_solver requires scipy (for scipy.optimize.minimize)."
    ) from exc

try:
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
except ImportError:
    from g1_kinematics import Pose
    from pinocchio_kinematics import (
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


class PinocchioDualArmSLSQPIKSolver:
    """Analytical-gradient SLSQP IK for the G1 dual arm, backed by Pinocchio.

    Parameters
    ----------
    urdf_path:
        URDF used to build the kinematic model. Defaults to the
        locked-waist, collision-simplified G1 URDF.
    max_iters:
        ``maxiter`` forwarded to ``scipy.optimize.minimize``. SLSQP
        converges on most targets in 5-15 major iters; 80 is a
        conservative cap.
    ftol:
        ``ftol`` forwarded to SLSQP: the solver stops when the
        objective cannot be improved by more than this amount
        between successive line searches. Default 1e-12 gets the IK
        residual down to ~0.01 mm / <0.01 deg, which is what you
        want for a publication comparison. Looser values converge
        faster but leave larger residuals (at 1e-10 you already see
        ~3 mm residuals in tracking because the initial objective
        is already below 1e-6 when the seed is the previous IK).
    tol_pos, tol_rot:
        Post-hoc success thresholds (meters / radians). These do
        NOT drive SLSQP's stopping criterion; they only set the
        ``success`` flag on the returned :class:`DualIKSolveResult`.
    pos_weight, rot_weight:
        Diagonal weights on the 6D error. Setting ``rot_weight`` <
        ``pos_weight`` prioritizes hitting the position target,
        which is usually desirable for upper-body teleop and
        matches the defaults of :class:`PinocchioDualArmDLSIKSolver`.
    reg_weight:
        Smoothness/stability regularization toward ``q_seed``. Small
        but non-zero (default 1e-6) helps SLSQP pick the IK branch
        closest to the current pose and keeps the QP sub-problem
        well-conditioned near singularities. Larger values (1e-4)
        bias the solution away from the exact IK by ~0.5 mm when
        the seed is far from the answer, so keep this small unless
        you are specifically fighting null-space drift in
        open-loop tracking. Set to 0 to disable.
    left_ee_frame, right_ee_frame, reference_frame, left_arm_joints,
    right_arm_joints:
        Passed through to :class:`G1DualArmPinocchioKinematics`.
    joint_limits:
        14x2 array of (lower, upper) joint limits used as box
        constraints. Defaults to :data:`DUAL_JOINT_LIMITS`.
    """

    def __init__(
        self,
        urdf_path: Optional[Path | str] = None,
        *,
        max_iters: int = 100,
        ftol: float = 1e-12,
        tol_pos: float = 5e-4,
        tol_rot: float = np.deg2rad(0.8),
        pos_weight: float = 1.0,
        rot_weight: float = 0.5,
        reg_weight: float = 1e-6,
        left_ee_frame: str = DEFAULT_LEFT_EE_FRAME,
        right_ee_frame: str = DEFAULT_RIGHT_EE_FRAME,
        reference_frame: str = DEFAULT_REFERENCE_FRAME,
        left_arm_joints: Sequence[str] = DEFAULT_LEFT_ARM_JOINTS,
        right_arm_joints: Sequence[str] = DEFAULT_RIGHT_ARM_JOINTS,
        joint_limits: Optional[np.ndarray] = None,
    ) -> None:
        self.kin = G1DualArmPinocchioKinematics(
            urdf_path=urdf_path,
            left_ee_frame=left_ee_frame,
            right_ee_frame=right_ee_frame,
            reference_frame=reference_frame,
            left_arm_joints=left_arm_joints,
            right_arm_joints=right_arm_joints,
        )
        self._pin = self.kin.pin

        self.urdf_path = self.kin.urdf_path
        self.reference_frame = self.kin.reference_frame
        self.nq = self.kin.nq
        self.nv = self.kin.nv
        self.left_ee_id = self.kin.left_ee_id
        self.right_ee_id = self.kin.right_ee_id

        self.limits = joint_limits if joint_limits is not None else DUAL_JOINT_LIMITS
        assert self.limits.shape == (14, 2)

        self.max_iters = int(max_iters)
        self.ftol = float(ftol)
        self.tol_pos = float(tol_pos)
        self.tol_rot = float(tol_rot)
        self.pos_weight = float(pos_weight)
        self.rot_weight = float(rot_weight)
        self.reg_weight = float(reg_weight)

        # Precomputed weighting matrices (diagonal stored as a vector
        # so we can fuse the W^T W multiply into an elementwise op).
        self._W6 = np.array(
            [self.pos_weight, self.pos_weight, self.pos_weight,
             self.rot_weight, self.rot_weight, self.rot_weight],
            dtype=np.float64,
        )
        self._W2_6 = self._W6 * self._W6  # diag(W^T W)

        self._bounds = Bounds(self.limits[:, 0].copy(), self.limits[:, 1].copy())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, target_left: Pose, target_right: Pose, seed_q14: np.ndarray
    ) -> DualIKSolveResult:
        """Return an IK solution that minimizes dual-arm pose error.

        Parameters
        ----------
        target_left, target_right:
            Targets in ``reference_frame`` (default: ``torso_link``).
        seed_q14:
            Initial guess and regularization anchor, concatenated as
            ``[left7, right7]``.
        """
        pin = self._pin
        seed = np.clip(
            np.asarray(seed_q14, dtype=np.float64).copy(),
            self.limits[:, 0], self.limits[:, 1],
        )

        p_tgt_L, R_tgt_L = self.kin.ref_to_root(target_left)
        p_tgt_R, R_tgt_R = self.kin.ref_to_root(target_right)

        W2 = self._W2_6
        reg = self.reg_weight

        # ---- cached FK + Jacobian evaluation -------------------------
        # SLSQP calls ``objective(q)`` and ``gradient(q)`` many times
        # with the same q during its line searches. Evaluate FK once
        # per unique q and memoize the weighted error and Jacobian.
        #
        # We compare against ``np.array_equal`` on the last-seen q
        # because scipy hands us brand-new arrays each call; a hash
        # would not work.
        cache = {"q": None, "eL": None, "eR": None, "JL": None, "JR": None}

        def _eval(q: np.ndarray) -> None:
            if cache["q"] is not None and np.array_equal(cache["q"], q):
                return
            p_L, R_L, p_R, R_R, J_L, J_R = self.kin.fk_and_jacobians(q)
            eL = np.empty(6, dtype=np.float64)
            eL[:3] = p_tgt_L - p_L
            eL[3:] = pin.log3(R_tgt_L @ R_L.T)
            eR = np.empty(6, dtype=np.float64)
            eR[:3] = p_tgt_R - p_R
            eR[3:] = pin.log3(R_tgt_R @ R_R.T)
            cache["q"] = q.copy()
            cache["eL"] = eL
            cache["eR"] = eR
            cache["JL"] = J_L
            cache["JR"] = J_R

        def objective(q: np.ndarray) -> float:
            _eval(q)
            eL, eR = cache["eL"], cache["eR"]
            # 0.5 * ||W e||^2 = 0.5 * sum(W^2 * e^2)
            cost = 0.5 * float(np.dot(W2 * eL, eL))
            cost += 0.5 * float(np.dot(W2 * eR, eR))
            if reg > 0.0:
                dq = q - seed
                cost += 0.5 * reg * float(np.dot(dq, dq))
            return cost

        def gradient(q: np.ndarray) -> np.ndarray:
            _eval(q)
            eL, eR = cache["eL"], cache["eR"]
            J_L, J_R = cache["JL"], cache["JR"]
            grad = np.zeros(14, dtype=np.float64)
            # d(0.5 ||W e||^2)/dq = -(J^T) (W^2 e)
            grad[:7] = -(J_L.T @ (W2 * eL))
            grad[7:] = -(J_R.T @ (W2 * eR))
            if reg > 0.0:
                grad += reg * (q - seed)
            return grad

        result = minimize(
            objective,
            seed,
            jac=gradient,
            method="SLSQP",
            bounds=self._bounds,
            options={
                "maxiter": self.max_iters,
                "ftol": self.ftol,
                "disp": False,
            },
        )

        # SLSQP respects ``Bounds`` strictly, but numerical noise can
        # land a coordinate 1e-16 outside; project back to be safe.
        q = np.clip(
            np.asarray(result.x, dtype=np.float64),
            self.limits[:, 0], self.limits[:, 1],
        )

        # Recompute residuals at the returned q (the cache may hold
        # a different intermediate q, and we always want to report
        # the honest residual at what we return).
        p_L, R_L, p_R, R_R, _, _ = self.kin.fk_and_jacobians(q)
        last_pL = float(np.linalg.norm(p_tgt_L - p_L))
        last_pR = float(np.linalg.norm(p_tgt_R - p_R))
        last_rL = float(np.linalg.norm(pin.log3(R_tgt_L @ R_L.T)))
        last_rR = float(np.linalg.norm(pin.log3(R_tgt_R @ R_R.T)))

        lp_deg = float(np.rad2deg(last_rL))
        rp_deg = float(np.rad2deg(last_rR))
        success = (
            last_pL < self.tol_pos
            and last_pR < self.tol_pos
            and last_rL < self.tol_rot
            and last_rR < self.tol_rot
        )

        return DualIKSolveResult(
            q=q,
            success=success,
            iterations=int(result.nit),
            left_pos_error_m=last_pL,
            right_pos_error_m=last_pR,
            left_rot_error_deg=lp_deg,
            right_rot_error_deg=rp_deg,
            pos_error_m=0.5 * (last_pL + last_pR),
            rot_error_deg=0.5 * (lp_deg + rp_deg),
        )

    def forward_kinematics(self, q14: np.ndarray) -> tuple[Pose, Pose]:
        """Pinocchio-backed FK in the same format as ``g1_kinematics``."""
        return self.kin.forward_kinematics(q14)


# ----------------------------------------------------------------------
# CLI: quick solve benchmark (same structure as pinocchio_dls_solver)
# ----------------------------------------------------------------------


def _cli_main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Pinocchio SLSQP dual-arm IK benchmark.")
    parser.add_argument("--urdf", type=Path, default=None, help="override URDF path (defaults to locked-waist G1)")
    parser.add_argument("--n-trials", type=int, default=200, help="number of random target trials")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--ftol", type=float, default=1e-12)
    parser.add_argument("--reg-weight", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    solver = PinocchioDualArmSLSQPIKSolver(
        urdf_path=args.urdf,
        max_iters=args.max_iters,
        ftol=args.ftol,
        reg_weight=args.reg_weight,
    )
    print(f"[pin-slsqp] URDF: {solver.urdf_path}")
    print(
        f"[pin-slsqp] model: nq={solver.nq}, nv={solver.nv}; "
        f"left_ee_id={solver.left_ee_id}, right_ee_id={solver.right_ee_id}; "
        f"reference_frame={solver.reference_frame}"
    )

    times_ms: list[float] = []
    iters_list: list[int] = []
    pos_errs: list[float] = []
    rot_errs: list[float] = []
    successes = 0
    for _ in range(args.n_trials):
        q_star = rng.uniform(low=DUAL_JOINT_LIMITS[:, 0], high=DUAL_JOINT_LIMITS[:, 1])
        target_L, target_R = solver.forward_kinematics(q_star)
        seed = q_star + rng.normal(scale=0.20, size=14)
        seed = np.clip(seed, DUAL_JOINT_LIMITS[:, 0], DUAL_JOINT_LIMITS[:, 1])

        t0 = time.perf_counter()
        result = solver.solve(target_L, target_R, seed)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        iters_list.append(result.iterations)
        pos_errs.append(result.pos_error_m)
        rot_errs.append(result.rot_error_deg)
        if result.success:
            successes += 1

    tt = np.array(times_ms)
    ie = np.array(iters_list)
    pe = np.array(pos_errs)
    re = np.array(rot_errs)
    print(
        f"[pin-slsqp] benchmark over {args.n_trials} random reachable targets:\n"
        f"  solve time : mean={tt.mean():.2f} ms, p50={np.median(tt):.2f}, p95={np.percentile(tt, 95):.2f}, max={tt.max():.2f}\n"
        f"  iterations : mean={ie.mean():.1f}, max={ie.max()}\n"
        f"  pos error  : mean={pe.mean()*1000:.3f} mm, max={pe.max()*1000:.3f} mm\n"
        f"  rot error  : mean={re.mean():.3f} deg, max={re.max():.3f} deg\n"
        f"  success    : {successes}/{args.n_trials} ({100.0 * successes / max(1, args.n_trials):.1f}%)"
    )


if __name__ == "__main__":
    _cli_main()
