"""Replay a bimanual wrist EE trajectory through the TWIST2 sim2sim policy.

Pipeline
--------

                               Redis (localhost:6379)
                                    |
      +------------------+         ---          +------------------------+
      | this script       | -----> | ----->     | server_low_level_g1_sim|
      | (publishes 35D    |  mimic_obs          |  (MuJoCo + ONNX policy) |
      |  mimic_obs to     |                     |                          |
      |  Redis)           | <-----            + |                          |
      +------------------+        ---           +------------------------+
                            state_body (34D)

This script is the drop-in replacement for ``xrobot_teleop_to_robot_w_hand.py``
when we want to test TWIST2 tracking on a recorded EE trajectory (instead
of a live VR teleop stream).

What it does
------------
1. Loads a bimanual wrist EE trajectory CSV (torso_link frame, 6D pos+RPY
   per hand). Format is the same as
   ``rl_ik_solver/trajectory/recorded_trajectories/traj1.csv``.
2. Offline-precomputes the 14-DoF arm joint reference via Pinocchio SLSQP
   (``rl_ik_solver/trajectory/ik_benchmark/pinocchio_slsqp_solver.py``),
   warm-started frame-to-frame so the solution stays on a single IK branch.
3. Waits for the TWIST2 sim server to come up (``state_body_*`` on Redis).
4. Reads the CURRENT measured arm joints from Redis, then does a
   synchronous joint-space ramp-in (default 2s) to ``q14[0]`` while
   publishing mimic_obs each tick, so the policy can drive the arms to
   the trajectory start pose smoothly. The clock is only started AFTER
   ramp-in finishes, so ``elapsed = 0`` aligns with the first real
   trajectory sample (identical convention to rl_ik_solver/trajectory.py).
5. Main loop at ``--control_frequency`` (default 50 Hz to match the
   TWIST2 training decimation and the rl_ik_solver baseline):
     - Sample target EE pose from the CSV at the current elapsed time.
     - Sample the precomputed q14 at the same time (same linear/
       wrap-aware interpolation).
     - Build the 35D mimic_obs:
         [vx=0, vy=0, z=0.793, roll=0, pitch=0, yaw_vel=0,
          default_dof_pos_with_arm_replaced(29)]
     - Publish to Redis key ``action_body_unitree_g1_with_hands``.
     - Read back (via one Redis pipeline):
         * sim state_body (34D: ang_vel(3)+rpy[:2](2)+dof_pos(29)),
         * the latest 29D PD target published by the sim server,
         * the per-step ONNX inference latency published by the sim
           server.
     - Compute the actual wrist pose in torso_link from the measured
       arm joints using the SAME analytical FK + pre-FK joint-limit
       clip that rl_ik_solver/.../trajectory.py uses, then feed the
       target (CSV) + actual (FK) poses to the TrackingErrorMonitor.
     - Feed the 14D arm slice of the PD target to the
       CommandSmoothnessMonitor, and the inference latency to the
       compute accumulator -- both metrics are then byte-identical to
       rl_ik_solver's in-process run.
6. On finish (or Ctrl-C) writes the same JSON/NPZ artifacts as
   ``rl_ik_solver/.../trajectory.py`` (same schema: tracking_error,
   smoothness, compute) so the two methods can be compared directly
   on the same trajectory.

Usage
-----
Two terminals, run from the TWIST2 repo root. The shell wrappers do
NOT activate any conda env -- activate the right one yourself first
in each terminal.

  # Terminal 1: sim server (needs mujoco + onnxruntime + torch + redis).
  conda activate gmr   # or whatever env has those deps on your host
  bash sim2sim_traj.sh

  # Terminal 2: trajectory-replay driver (needs pinocchio + scipy +
  # redis + numpy). Defaults to --control_frequency 50 (matches
  # Terminal 1) and uses the vendored sample CSV
  # ``assets/trajectories/traj1.csv``.
  conda activate env_isaacgym
  bash sim2sim_traj_replay.sh

  # Custom trajectory / IK tuning (the shell forwards extra flags to
  # trajectory_replay_sim2sim.py via "$@"):
  TRAJECTORY_CSV=/path/to/your.csv bash sim2sim_traj_replay.sh --ramp_in_sec 2.0

  # Or call python directly, skipping the wrapper:
  cd deploy_real
  python trajectory_replay_sim2sim.py \
      --trajectory_csv ../assets/trajectories/traj1.csv \
      --control_frequency 50 \
      --ramp_in_sec 2.0

Artifacts (JSON summary + NPZ trace + PNG figure) are written to
``<TWIST2>/tracking_results/<csv_stem>_<timestamp>.*`` by default;
override with ``--result_json``.

Dependencies
------------
* numpy
* scipy (for SLSQP)
* pinocchio (via rl_ik_solver's _import_pinocchio_safely helper)
* redis (for IPC with the sim server)
* loop_rate_limiters (same rate-limiter class used by teleop)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import redis


# ``<TWIST2>/deploy_real/trajectory_replay_sim2sim.py`` -> parents[1] is <TWIST2>.
# Every default path below is resolved relative to this so the repo is
# self-contained (no hard-coded absolute paths into sibling repos).
TWIST2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_CSV = TWIST2_ROOT / "assets" / "trajectories" / "traj1.csv"
DEFAULT_RESULTS_DIR = TWIST2_ROOT / "tracking_results"

try:
    from loop_rate_limiters import RateLimiter as _ExternalRateLimiter
    _HAVE_EXTERNAL_RATE = True
except ImportError:
    _HAVE_EXTERNAL_RATE = False


class _FallbackRateLimiter:
    """Minimal stand-in for ``loop_rate_limiters.RateLimiter``.

    Not strictly drift-free (best-effort ``time.sleep`` to the next tick),
    but good enough for this replay path where the sim server already
    runs its own loop at the same nominal frequency. Dependencies: none.
    """

    def __init__(self, frequency: float, warn: bool = False) -> None:
        self._dt = 1.0 / max(float(frequency), 1e-6)
        self._next = time.monotonic() + self._dt

    def sleep(self) -> None:
        now = time.monotonic()
        delay = self._next - now
        if delay > 0.0:
            time.sleep(delay)
            self._next += self._dt
        else:
            # We overran; reset the schedule so we don't spiral.
            self._next = time.monotonic() + self._dt


def _make_rate_limiter(frequency: float):
    if _HAVE_EXTERNAL_RATE:
        return _ExternalRateLimiter(frequency=frequency, warn=False)
    return _FallbackRateLimiter(frequency=frequency)

# Reuse the exact IK stack that rl_ik_solver's ik_benchmark uses, so
# the joint reference we feed TWIST2 and the rl_ik_solver baseline see
# the same kinematic model (torso_link reference, wrist_yaw_link EE).
# The three modules under ``ik_benchmark/`` are vendored copies of
# rl_ik_solver/trajectory/ik_benchmark/*.py -- see ik_benchmark/__init__.py.
from ik_benchmark import (  # type: ignore
    Pose,
    PinocchioDualArmSLSQPIKSolver,
)

from tracking_metrics import (
    CommandSmoothnessMonitor,
    TrackingErrorMonitor,
    TrajectorySampler,
    rpy_to_quaternion_wxyz,
)

# Analytical FK + joint-limit clip copied verbatim from rl_ik_solver so
# the "actual EE" pose fed into TrackingErrorMonitor is bit-identical to
# the rl_ik_solver baseline (same formula, same pre-FK clip).
from g1_analytical_kinematics import (
    LEFT_JOINT_LIMITS_RAD,
    RIGHT_JOINT_LIMITS_RAD,
    larm_forward,
    rarm_forward,
)


# ----------------------------------------------------------------------
# TWIST2 G1 29-DoF joint layout (matches server_low_level_g1_sim.py)
# ----------------------------------------------------------------------
# 0-5   left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# 6-11  right leg
# 12-14 waist (yaw, roll, pitch)
# 15-21 left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#                 wrist_roll, wrist_pitch, wrist_yaw)
# 22-28 right arm

LEFT_ARM_SLOT = np.arange(15, 22, dtype=np.int64)
RIGHT_ARM_SLOT = np.arange(22, 29, dtype=np.int64)

# Copied from server_low_level_g1_sim.py to stay in sync.
DEFAULT_DOF_POS = np.array(
    [
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Default standing state used as the non-arm part of the 35D mimic obs
# (vx, vy, z, roll, pitch, yaw_vel). Matches the TWIST2 teleop defaults
# for a static standing pose.
DEFAULT_BASE_OBS6 = np.array(
    [0.0, 0.0, 0.793, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


# ----------------------------------------------------------------------
# Redis keys (matches server_low_level_g1_sim.py)
# ----------------------------------------------------------------------

K_ACTION_BODY = "action_body_unitree_g1_with_hands"
K_ACTION_HAND_L = "action_hand_left_unitree_g1_with_hands"
K_ACTION_HAND_R = "action_hand_right_unitree_g1_with_hands"
K_ACTION_NECK = "action_neck_unitree_g1_with_hands"
K_STATE_BODY = "state_body_unitree_g1_with_hands"
K_POLICY_PD_TARGET = "policy_pd_target_unitree_g1_with_hands"
K_POLICY_INFER_MS = "policy_infer_ms_unitree_g1_with_hands"
K_T_ACTION = "t_action"


# ----------------------------------------------------------------------
# Main replay controller
# ----------------------------------------------------------------------


class TrajectoryReplaySim2Sim:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dt = 1.0 / float(args.control_frequency)
        self.loop_trajectory = bool(args.loop_trajectory)
        self.auto_stop_on_finish = not bool(args.disable_auto_stop_on_finish)
        self.ramp_in_sec = max(0.0, float(args.ramp_in_sec))

        # --- Redis -----------------------------------------------------
        self.redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print(f"[Redis] connected at {args.redis_ip}:{args.redis_port}")

        # --- Trajectory + IK ------------------------------------------
        csv_path = Path(args.trajectory_csv).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")
        self.traj = TrajectorySampler(csv_path)
        print(
            f"[Trajectory] loaded {csv_path.name}: "
            f"{self.traj.sample_count} samples, {self.traj.duration:.2f}s."
        )

        # The online actual-EE pose uses the analytical FK from
        # rl_ik_solver (see g1_analytical_kinematics.py), so we no
        # longer need a Pinocchio FK handle here. The IK solver below
        # carries its own internal Pinocchio state.
        self.ik_solver = PinocchioDualArmSLSQPIKSolver(
            urdf_path=args.urdf_path if args.urdf_path else None,
            max_iters=int(args.ik_max_iters),
            ftol=float(args.ik_ftol),
            pos_weight=1.0,
            rot_weight=0.5,
            reg_weight=float(args.ik_reg_weight),
        )

        self.q14_offline = self._precompute_ik(self.traj)

        # --- Metric monitors ------------------------------------------
        error_log_csv = (
            Path(args.error_log_csv).expanduser().resolve()
            if args.error_log_csv
            else None
        )
        self.error_monitor = (
            None
            if args.disable_tracking_error
            else TrackingErrorMonitor(
                print_hz=float(args.error_print_hz),
                log_csv_path=error_log_csv,
                warmup_sec=float(args.error_warmup_sec),
                lag_search_max_sec=float(args.error_lag_search_sec),
                spatial_resample_count=int(args.error_spatial_resample_count),
            )
        )
        self.smoothness_monitor = (
            None
            if args.disable_smoothness_metric
            else CommandSmoothnessMonitor(
                print_hz=float(args.smoothness_print_hz),
                warmup_sec=float(args.smoothness_warmup_sec),
                fixed_dt=self.dt,
            )
        )

        # Buffers for NPZ dump (kept separate from the monitor, which
        # already stores target+actual pose samples; we only need the
        # joint-command trace here).
        self.joint_cmd_times: list[float] = []
        self.joint_cmd_samples: list[np.ndarray] = []

        # Per-step ONNX inference latency (ms) as published by the sim
        # server. Only filled after warmup, matching rl_ik_solver's
        # infer_times_ms semantics.
        self.infer_times_ms: list[float] = []
        self._last_seen_pd_target: np.ndarray | None = None
        self._last_seen_infer_ms: float | None = None

        # --- Default hand / neck actions ------------------------------
        self.default_hand = np.zeros(7, dtype=np.float64)
        self.default_neck = np.array([0.0, 0.0], dtype=np.float64)

        # --- Runtime state --------------------------------------------
        self.control_start_time: float | None = None
        self.stop_requested = False
        self.stop_reason = "running"
        self.trajectory_finished = False

    # ------------------------------------------------------------------
    # Offline IK precompute
    # ------------------------------------------------------------------

    def _precompute_ik(self, traj: TrajectorySampler) -> np.ndarray:
        print(f"[IK] precomputing SLSQP solutions for {traj.sample_count} frames...")
        t0 = time.perf_counter()
        q14 = np.zeros((traj.sample_count, 14), dtype=np.float64)
        seed = np.zeros(14, dtype=np.float64)
        max_pos_err = 0.0
        max_rot_err_deg = 0.0
        n_fail = 0
        for i in range(traj.sample_count):
            l_rpy = traj.left_traj[i, 3:6]
            r_rpy = traj.right_traj[i, 3:6]
            l_pose = Pose(
                pos=traj.left_traj[i, :3].astype(np.float64),
                quat_wxyz=rpy_to_quaternion_wxyz(l_rpy),
            )
            r_pose = Pose(
                pos=traj.right_traj[i, :3].astype(np.float64),
                quat_wxyz=rpy_to_quaternion_wxyz(r_rpy),
            )
            result = self.ik_solver.solve(l_pose, r_pose, seed)
            q14[i] = result.q
            seed = result.q
            pos_err = max(result.left_pos_error_m, result.right_pos_error_m)
            rot_err = max(result.left_rot_error_deg, result.right_rot_error_deg)
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err_deg = max(max_rot_err_deg, rot_err)
            if not result.success:
                n_fail += 1
        elapsed = time.perf_counter() - t0
        print(
            f"[IK] done in {elapsed:.2f}s. "
            f"max residual pos={max_pos_err*1000:.2f}mm rot={max_rot_err_deg:.2f}deg. "
            f"fail_count={n_fail}/{traj.sample_count}"
        )
        return q14

    def _sample_q14(self, elapsed: float) -> np.ndarray:
        """Linear interpolation of the offline IK q14 at a given elapsed time."""
        ts = self.traj.timestamps
        if self.loop_trajectory and self.traj.duration > 0.0:
            elapsed = elapsed % self.traj.duration
        else:
            elapsed = float(np.clip(elapsed, 0.0, self.traj.duration))
        upper = int(np.searchsorted(ts, elapsed, side="right"))
        if upper <= 0:
            return self.q14_offline[0].copy()
        if upper >= self.traj.sample_count:
            return self.q14_offline[-1].copy()
        lower = upper - 1
        t0, t1 = float(ts[lower]), float(ts[upper])
        alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
        return (1.0 - alpha) * self.q14_offline[lower] + alpha * self.q14_offline[upper]

    # ------------------------------------------------------------------
    # Redis I/O
    # ------------------------------------------------------------------

    def _read_state_body(self) -> np.ndarray | None:
        """Return the 34-D state_body from Redis, or None if unavailable."""
        raw = self.redis_client.get(K_STATE_BODY)
        if raw is None:
            return None
        try:
            arr = np.asarray(json.loads(raw), dtype=np.float64)
        except (ValueError, TypeError):
            return None
        # During the sim server's bootstrap, it publishes a 127-D zero
        # vector; wait for the 34-D payload that the main loop writes.
        if arr.size != 34:
            return None
        return arr

    def _read_sim_outputs(self):
        """Pipeline-read (state_body, pd_target_29D, infer_ms) from the sim server.

        Returns
        -------
        (state_body | None, pd_target | None, infer_ms | None)

        Any value may be None if the corresponding Redis key is missing
        or cannot be parsed. The sim server publishes ``pd_target`` and
        ``infer_ms`` once per policy tick right next to the
        state; reading them in one pipeline minimises jitter.
        """
        self.redis_pipeline.get(K_STATE_BODY)
        self.redis_pipeline.get(K_POLICY_PD_TARGET)
        self.redis_pipeline.get(K_POLICY_INFER_MS)
        results = self.redis_pipeline.execute()

        raw_state, raw_pd, raw_infer = results
        state = None
        if raw_state is not None:
            try:
                arr = np.asarray(json.loads(raw_state), dtype=np.float64)
                if arr.size == 34:
                    state = arr
            except (ValueError, TypeError):
                pass

        pd_target = None
        if raw_pd is not None:
            try:
                arr = np.asarray(json.loads(raw_pd), dtype=np.float64)
                if arr.size == 29:
                    pd_target = arr
            except (ValueError, TypeError):
                pass

        infer_ms = None
        if raw_infer is not None:
            try:
                infer_ms = float(raw_infer)
            except (ValueError, TypeError):
                pass

        return state, pd_target, infer_ms

    def _wait_for_sim_state(self, timeout_sec: float = 30.0) -> np.ndarray:
        print("[Replay] waiting for sim server state_body (34D) ...")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            state = self._read_state_body()
            if state is not None:
                print("[Replay] sim state_body is live.")
                return state
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for TWIST2 sim server. "
            "Make sure sim2sim.sh is running (server_low_level_g1_sim.py)."
        )

    def _build_mimic_obs(self, q14: np.ndarray) -> np.ndarray:
        dof_pos = DEFAULT_DOF_POS.copy()
        dof_pos[LEFT_ARM_SLOT] = q14[:7]
        dof_pos[RIGHT_ARM_SLOT] = q14[7:]
        return np.concatenate([DEFAULT_BASE_OBS6, dof_pos]).astype(np.float64)

    def _publish(self, mimic_obs: np.ndarray) -> None:
        assert mimic_obs.size == 35, f"expected 35D mimic_obs, got {mimic_obs.size}"
        self.redis_pipeline.set(K_ACTION_BODY, json.dumps(mimic_obs.tolist()))
        self.redis_pipeline.set(K_ACTION_HAND_L, json.dumps(self.default_hand.tolist()))
        self.redis_pipeline.set(K_ACTION_HAND_R, json.dumps(self.default_hand.tolist()))
        self.redis_pipeline.set(K_ACTION_NECK, json.dumps(self.default_neck.tolist()))
        self.redis_pipeline.set(K_T_ACTION, int(time.time() * 1000))
        self.redis_pipeline.execute()

    # ------------------------------------------------------------------
    # FK for actual EE pose (torso_link frame)
    # ------------------------------------------------------------------

    def _fk_actual(self, q14: np.ndarray):
        """Return (left_pose7, right_pose7) in torso_link frame as [pos(3), quat_wxyz(4)].

        Uses the analytical FK + pre-FK joint-limit clip copied verbatim
        from rl_ik_solver/.../trajectory.py so the "actual EE" pose fed
        into the TrackingErrorMonitor matches the rl_ik_solver baseline
        to machine precision.
        """
        q_arr = np.asarray(q14, dtype=np.float32)
        left_theta = np.clip(
            q_arr[:7], LEFT_JOINT_LIMITS_RAD[:, 0], LEFT_JOINT_LIMITS_RAD[:, 1]
        )
        right_theta = np.clip(
            q_arr[7:], RIGHT_JOINT_LIMITS_RAD[:, 0], RIGHT_JOINT_LIMITS_RAD[:, 1]
        )
        left_pose = larm_forward(left_theta).astype(np.float64)
        right_pose = rarm_forward(right_theta).astype(np.float64)
        return left_pose, right_pose

    # ------------------------------------------------------------------
    # Ramp-in (joint space)
    # ------------------------------------------------------------------

    def _ramp_in(self, rate) -> None:
        if self.ramp_in_sec <= 0.0:
            return
        print(f"[Ramp] ramping arm joints to trajectory[0] over {self.ramp_in_sec:.2f}s ...")
        state = self._read_state_body()
        if state is None:
            state = self._wait_for_sim_state()
        # dof_pos starts at index 5 in the 34D state_body
        # (3 ang_vel + 2 rpy + 29 dof_pos).
        dof_pos_measured = state[5:5 + 29]
        q14_start = np.concatenate(
            [dof_pos_measured[LEFT_ARM_SLOT], dof_pos_measured[RIGHT_ARM_SLOT]]
        )
        q14_target = self.q14_offline[0].copy()
        num_step = max(1, int(round(self.ramp_in_sec / self.dt)))
        for step in range(num_step):
            alpha = float(step + 1) / float(num_step)
            q14_interp = (1.0 - alpha) * q14_start + alpha * q14_target
            mimic_obs = self._build_mimic_obs(q14_interp)
            self._publish(mimic_obs)
            rate.sleep()
        print("[Ramp] complete.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        rate = _make_rate_limiter(float(self.args.control_frequency))

        # Wait for sim, then ramp-in.
        self._wait_for_sim_state()
        self._ramp_in(rate)

        self.control_start_time = time.monotonic()
        print(
            f"[Replay] start. duration={self.traj.duration:.2f}s "
            f"loop={self.loop_trajectory} auto_stop={self.auto_stop_on_finish}"
        )

        try:
            while True:
                elapsed = time.monotonic() - self.control_start_time

                # Trajectory finished?
                if (
                    not self.loop_trajectory
                    and not self.trajectory_finished
                    and elapsed >= self.traj.duration
                ):
                    self.trajectory_finished = True
                    print("[Replay] trajectory playback completed.")
                    if self.auto_stop_on_finish:
                        self.stop_requested = True
                        self.stop_reason = "trajectory_finished"

                if self.stop_requested:
                    break

                # Sample target EE (CSV) + reference q14 (IK precomputed).
                left_target_6, right_target_6 = self.traj.sample(
                    elapsed, loop=self.loop_trajectory
                )
                q14_ref = self._sample_q14(elapsed)

                # Publish mimic_obs.
                mimic_obs = self._build_mimic_obs(q14_ref)
                self._publish(mimic_obs)

                # Read sim state + latest policy PD target + inference
                # time in one pipeline round-trip.
                state, pd_target, infer_ms = self._read_sim_outputs()

                # Cache the most recent pd_target / infer_ms so that a
                # transiently missing key doesn't poison the summary.
                if pd_target is not None:
                    self._last_seen_pd_target = pd_target
                if infer_ms is not None:
                    self._last_seen_infer_ms = infer_ms

                if state is not None:
                    dof_pos_measured = state[5:5 + 29]
                    q14_measured = np.concatenate(
                        [dof_pos_measured[LEFT_ARM_SLOT], dof_pos_measured[RIGHT_ARM_SLOT]]
                    )
                    left_actual_7, right_actual_7 = self._fk_actual(q14_measured)

                    if self.error_monitor is not None:
                        self.error_monitor.update(
                            elapsed_time=elapsed,
                            left_target_euler=left_target_6,
                            right_target_euler=right_target_6,
                            left_actual_pose=left_actual_7,
                            right_actual_pose=right_actual_7,
                        )

                # Smoothness / compute / joint_cmd trace all use the
                # POLICY OUTPUT (pd_target), 14D arm subset, so they
                # match the rl_ik_solver baseline byte-for-byte.
                if self._last_seen_pd_target is not None:
                    arm_cmd_14 = np.concatenate(
                        [
                            self._last_seen_pd_target[LEFT_ARM_SLOT],
                            self._last_seen_pd_target[RIGHT_ARM_SLOT],
                        ]
                    )
                    if self.smoothness_monitor is not None:
                        self.smoothness_monitor.update(elapsed, arm_cmd_14)

                    if elapsed >= float(self.args.error_warmup_sec):
                        self.joint_cmd_times.append(float(elapsed))
                        self.joint_cmd_samples.append(arm_cmd_14.copy())

                if (
                    self._last_seen_infer_ms is not None
                    and elapsed >= float(self.args.error_warmup_sec)
                ):
                    self.infer_times_ms.append(float(self._last_seen_infer_ms))
                    # Only consume each sample once to avoid double-counting
                    # when the sim server lags behind our control loop.
                    self._last_seen_infer_ms = None

                rate.sleep()

        except KeyboardInterrupt:
            self.stop_reason = "keyboard_interrupt"
            print("[Replay] Ctrl-C received, stopping.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown (send one last "hold" obs, then dump artifacts)
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        # Keep publishing a static hold pose so the policy doesn't see
        # stale future commands; use the LAST q14 as the hold target.
        try:
            q14_hold = self.q14_offline[-1]
            mimic_obs = self._build_mimic_obs(q14_hold)
            self._publish(mimic_obs)
        except Exception:
            pass

        if self.error_monitor is not None:
            self.error_monitor.print_summary()
            self.error_monitor.close()
        if self.smoothness_monitor is not None:
            self.smoothness_monitor.print_summary()

        result_summary = self._build_result_summary()
        result_json_path = self._resolve_result_json_path()
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        with result_json_path.open("w", encoding="ascii") as f:
            json.dump(result_summary, f, indent=2)
        print(f"[Replay] saved run summary to: {result_json_path}")

        if not self.args.disable_plot:
            trace_path = result_json_path.with_suffix(".npz")
            figure_path = result_json_path.with_suffix(".png")
            if self._save_trace(trace_path):
                print(f"[Replay] saved trajectory trace:  {trace_path}")
                self._try_plot(trace_path, figure_path)
            else:
                print("[Replay] no trace samples available; skipping figure.")

    def _resolve_result_json_path(self) -> Path:
        if self.args.result_json:
            return Path(self.args.result_json).expanduser().resolve()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_stem = Path(self.args.trajectory_csv).stem
        return DEFAULT_RESULTS_DIR / f"{csv_stem}_{ts}.json"

    def _build_result_summary(self) -> dict:
        elapsed = 0.0
        if self.control_start_time is not None:
            elapsed = max(0.0, time.monotonic() - self.control_start_time)

        if self.infer_times_ms:
            infer_arr = np.asarray(self.infer_times_ms, dtype=np.float64)
            compute_summary = {
                "sample_count": int(infer_arr.size),
                "time_ms_mean": float(infer_arr.mean()),
                "time_ms_std": float(infer_arr.std(ddof=0)),
                "time_ms_p50": float(np.percentile(infer_arr, 50)),
                "time_ms_p95": float(np.percentile(infer_arr, 95)),
                "time_ms_max": float(infer_arr.max()),
            }
        else:
            compute_summary = {
                "sample_count": 0,
                "time_ms_mean": None,
                "time_ms_std": None,
                "time_ms_p50": None,
                "time_ms_p95": None,
                "time_ms_max": None,
            }

        # Schema matches rl_ik_solver/.../trajectory.py::build_result_summary
        # byte-for-byte (same field names, same order) so existing
        # comparison / plotting tools can consume both runs without any
        # code change. The TWIST2-specific metadata (IK tuning, the fact
        # that the policy is twist2) is stuffed into additive fields
        # that the rl_ik_solver tooling simply ignores.
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runtime_type": "rl",
            "method": "twist2_sim2sim",
            "trajectory_csv": str(Path(self.args.trajectory_csv).expanduser().resolve()),
            "trajectory_name": Path(self.args.trajectory_csv).stem,
            "loop_trajectory": self.loop_trajectory,
            "auto_stop_on_finish": self.auto_stop_on_finish,
            "trajectory_duration_sec": self.traj.duration,
            "control_dt_sec": self.dt,
            "ramp_in_sec": self.ramp_in_sec,
            "trajectory_sample_count": self.traj.sample_count,
            "elapsed_control_sec": elapsed,
            "trajectory_finished": self.trajectory_finished,
            "stop_reason": self.stop_reason,
            "tracking_error": (
                self.error_monitor.get_summary() if self.error_monitor else None
            ),
            "smoothness": (
                self.smoothness_monitor.get_summary() if self.smoothness_monitor else None
            ),
            "compute": compute_summary,
            "ik": {
                "solver": "pinocchio_slsqp",
                "max_iters": int(self.args.ik_max_iters),
                "ftol": float(self.args.ik_ftol),
                "reg_weight": float(self.args.ik_reg_weight),
            },
        }

    def _save_trace(self, trace_path: Path) -> bool:
        em = self.error_monitor
        if em is None or em.count == 0:
            return False
        times = np.asarray(em.sample_times, dtype=np.float64)
        lp = np.stack(em.left_target_pos_samples, axis=0).astype(np.float64)
        rp = np.stack(em.right_target_pos_samples, axis=0).astype(np.float64)
        lq = np.stack(em.left_target_quat_samples, axis=0).astype(np.float64)
        rq = np.stack(em.right_target_quat_samples, axis=0).astype(np.float64)
        lap = np.stack(em.left_actual_pose_samples, axis=0).astype(np.float64)
        rap = np.stack(em.right_actual_pose_samples, axis=0).astype(np.float64)
        left_target = np.concatenate([lp, lq], axis=1)
        right_target = np.concatenate([rp, rq], axis=1)
        if self.joint_cmd_samples:
            joint_cmd = np.stack(self.joint_cmd_samples, axis=0).astype(np.float64)
            joint_cmd_t = np.asarray(self.joint_cmd_times, dtype=np.float64)
        else:
            # 14D (7 left arm + 7 right arm), identical shape to the
            # rl_ik_solver baseline trace so plot_tracking_trace.py can
            # consume both runs without branching on runtime_type.
            joint_cmd = np.zeros((0, 14), dtype=np.float64)
            joint_cmd_t = np.zeros((0,), dtype=np.float64)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            t=times,
            left_target=left_target,
            right_target=right_target,
            left_actual=lap,
            right_actual=rap,
            joint_cmd=joint_cmd,
            joint_cmd_t=joint_cmd_t,
        )
        return True

    def _try_plot(self, trace_path: Path, figure_path: Path) -> None:
        try:
            # plot_tracking_trace.py sits next to this script under
            # deploy_real/, vendored from rl_ik_solver/trajectory/.
            from plot_tracking_trace import plot_trace  # type: ignore

            plot_trace(trace_path, figure_path)
            print(f"[Replay] saved tracking figure:   {figure_path}")
        except Exception as e:
            print(f"[Replay] figure generation failed: {e}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--trajectory_csv",
        type=str,
        default=str(DEFAULT_TRAJECTORY_CSV),
        help=(
            "Bimanual wrist EE trajectory CSV (torso_link frame, 6D pos+RPY "
            "per hand). Default is the vendored sample at "
            "assets/trajectories/traj1.csv."
        ),
    )
    p.add_argument(
        "--redis_ip", type=str, default="localhost",
        help="Redis server used for IPC with the TWIST2 sim server.",
    )
    p.add_argument("--redis_port", type=int, default=6379)
    p.add_argument(
        "--control_frequency", type=float, default=50.0,
        help=(
            "Replay publish frequency in Hz. Must match the sim server's "
            "--policy_frequency. Default is 50 Hz to stay in-distribution "
            "with the TWIST2 policy (training decimation is 50 Hz: "
            "legged_gym humanoid_config.py has sim dt=0.005 * decimation=4) "
            "and to match the rl_ik_solver baseline control_dt=0.02."
        ),
    )
    p.add_argument(
        "--loop_trajectory", action="store_true",
        help="Loop the CSV trajectory instead of holding the final pose.",
    )
    p.add_argument(
        "--disable_auto_stop_on_finish", action="store_true",
        help="Keep running after the trajectory finishes (default: auto stop).",
    )
    p.add_argument(
        "--ramp_in_sec", type=float, default=2.0,
        help=(
            "Synchronous joint-space ramp-in from the current measured arm pose "
            "to trajectory[0] before the main tracking loop starts. Set to 0 to "
            "disable (and consider --error_warmup_sec 0.5 to hide the transient)."
        ),
    )
    # IK tuning
    p.add_argument(
        "--urdf_path", type=str, default="",
        help="Override URDF path (default: the one embedded in pinocchio_kinematics.py).",
    )
    p.add_argument("--ik_max_iters", type=int, default=200)
    p.add_argument("--ik_ftol", type=float, default=1e-12)
    p.add_argument("--ik_reg_weight", type=float, default=1e-6)
    # Tracking-error monitor
    p.add_argument("--disable_tracking_error", action="store_true")
    p.add_argument("--error_print_hz", type=float, default=2.0)
    p.add_argument("--error_log_csv", type=str, default="")
    p.add_argument("--error_warmup_sec", type=float, default=0.0)
    p.add_argument("--error_lag_search_sec", type=float, default=1.0)
    p.add_argument("--error_spatial_resample_count", type=int, default=200)
    # Smoothness monitor
    p.add_argument("--disable_smoothness_metric", action="store_true")
    p.add_argument("--smoothness_print_hz", type=float, default=2.0)
    p.add_argument("--smoothness_warmup_sec", type=float, default=0.0)
    # Output
    p.add_argument(
        "--result_json", type=str, default="",
        help=(
            "Path to save the run-summary JSON. Default: "
            "<TWIST2>/tracking_results/<csv_stem>_<ts>.json"
        ),
    )
    p.add_argument("--disable_plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    replay = TrajectoryReplaySim2Sim(args)
    replay.run()


if __name__ == "__main__":
    main()
