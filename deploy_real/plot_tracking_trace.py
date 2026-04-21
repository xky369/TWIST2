#!/usr/bin/env python3
"""Plot end-effector tracking traces saved by trajectory.py / ik_controller.py.

Reads an NPZ trace file with the following arrays (all shape indicated):
    t              : (N,)      sample times in seconds
    left_target    : (N, 7)    target EE pose [x, y, z, qw, qx, qy, qz]
    right_target   : (N, 7)
    left_actual    : (N, 7)    measured EE pose from FK of actual joints
    right_actual   : (N, 7)
    joint_cmd      : (M, 14)   commanded arm joint positions (left7 + right7)
    joint_cmd_t    : (M,)      times for joint_cmd

Optional JSON sidecar <stem>.json is used for the figure title when available.

Usage:
    python3 trajectory/plot_tracking_trace.py --trace PATH.npz
    python3 trajectory/plot_tracking_trace.py --trace PATH.npz --output OUT.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/server friendly
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _quat_angle_error_batch(ref_wxyz: np.ndarray, act_wxyz: np.ndarray) -> np.ndarray:
    ref = ref_wxyz / np.clip(np.linalg.norm(ref_wxyz, axis=1, keepdims=True), 1e-12, None)
    act = act_wxyz / np.clip(np.linalg.norm(act_wxyz, axis=1, keepdims=True), 1e-12, None)
    cos_half = np.clip(np.abs(np.sum(ref * act, axis=1)), -1.0, 1.0)
    return 2.0 * np.arccos(cos_half)


def _title_from_sidecar(trace_path: Path) -> str:
    sidecar = trace_path.with_suffix(".json")
    if not sidecar.is_file():
        return trace_path.stem
    try:
        data = json.loads(sidecar.read_text(encoding="ascii"))
    except Exception:
        return trace_path.stem
    runtime_type = str(data.get("runtime_type") or "?")
    method = str(data.get("ik_method") or "-")
    traj = str(data.get("trajectory_name") or trace_path.stem)
    created = str(data.get("created_at") or "").replace("T", " ")[:19]
    suffix = f" ({runtime_type}" + (f"/{method}" if method != "-" else "") + ")"
    return f"{traj}{suffix} @ {created}" if created else f"{traj}{suffix}"


def plot_trace(trace_path: Path, output_path: Path | None = None) -> Path:
    data = np.load(trace_path)
    t = np.asarray(data["t"], dtype=np.float64)
    lt = np.asarray(data["left_target"], dtype=np.float64)
    rt = np.asarray(data["right_target"], dtype=np.float64)
    la = np.asarray(data["left_actual"], dtype=np.float64)
    ra = np.asarray(data["right_actual"], dtype=np.float64)
    has_joint = "joint_cmd" in data.files and data["joint_cmd"].size > 0
    joint_cmd = np.asarray(data["joint_cmd"], dtype=np.float64) if has_joint else None
    joint_cmd_t = (
        np.asarray(data["joint_cmd_t"], dtype=np.float64)
        if (has_joint and "joint_cmd_t" in data.files and data["joint_cmd_t"].size == joint_cmd.shape[0])
        else (np.arange(joint_cmd.shape[0]) if has_joint else None)
    )

    # Derived quantities
    l_pos_err = np.linalg.norm(lt[:, :3] - la[:, :3], axis=1)
    r_pos_err = np.linalg.norm(rt[:, :3] - ra[:, :3], axis=1)
    l_rot_err_deg = np.rad2deg(_quat_angle_error_batch(lt[:, 3:7], la[:, 3:7]))
    r_rot_err_deg = np.rad2deg(_quat_angle_error_batch(rt[:, 3:7], ra[:, 3:7]))

    if output_path is None:
        output_path = trace_path.with_suffix(".png")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 6, hspace=0.55, wspace=0.55)

    # Row 1: two 3D plots (target vs actual), spanning 3 cols each.
    for col, (pos_t, pos_a, tag, color_t, color_a) in enumerate(
        [
            (lt[:, :3], la[:, :3], "left arm", "tab:blue", "tab:orange"),
            (rt[:, :3], ra[:, :3], "right arm", "tab:blue", "tab:orange"),
        ]
    ):
        ax = fig.add_subplot(gs[0, col * 3 : (col + 1) * 3], projection="3d")
        ax.plot(pos_t[:, 0], pos_t[:, 1], pos_t[:, 2], color=color_t, label="target", linewidth=1.5)
        ax.plot(pos_a[:, 0], pos_a[:, 1], pos_a[:, 2], color=color_a, label="actual", linewidth=1.2, alpha=0.9)
        ax.scatter(pos_t[0, 0], pos_t[0, 1], pos_t[0, 2], c="green", s=20, label="start")
        ax.scatter(pos_t[-1, 0], pos_t[-1, 1], pos_t[-1, 2], c="red", s=20, label="end")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"3D EE trajectory - {tag}")
        ax.legend(loc="upper right", fontsize=8)

    # Row 2: per-axis position over time (6 subplots, 2 rows of 3 components for L/R).
    axis_labels = ["x", "y", "z"]
    for r_idx, (pos_t, pos_a, tag) in enumerate(
        [(lt[:, :3], la[:, :3], "L"), (rt[:, :3], ra[:, :3], "R")]
    ):
        for c_idx, axn in enumerate(axis_labels):
            ax = fig.add_subplot(gs[1, r_idx * 3 + c_idx])
            ax.plot(t, pos_t[:, c_idx], label="target", color="tab:blue", linewidth=1.2)
            ax.plot(t, pos_a[:, c_idx], label="actual", color="tab:orange", linewidth=1.0, alpha=0.9)
            ax.set_title(f"{tag}_{axn}(t) [m]")
            ax.set_xlabel("t [s]")
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=8, loc="best")

    # Row 3: position error and rotation error over time.
    ax = fig.add_subplot(gs[2, 0:3])
    ax.plot(t, l_pos_err, label="L pos err", color="tab:blue")
    ax.plot(t, r_pos_err, label="R pos err", color="tab:orange")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||p_tgt - p_act|| [m]")
    ax.set_title("Position tracking error")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 3:6])
    ax.plot(t, l_rot_err_deg, label="L rot err", color="tab:blue")
    ax.plot(t, r_rot_err_deg, label="R rot err", color="tab:orange")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("rotation angle error [deg]")
    ax.set_title("Orientation tracking error")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Row 4: joint commands over time (two subplots: left7 / right7).
    ax_l = fig.add_subplot(gs[3, 0:3])
    ax_r = fig.add_subplot(gs[3, 3:6])
    if has_joint and joint_cmd is not None:
        for j in range(7):
            ax_l.plot(joint_cmd_t, joint_cmd[:, j], linewidth=0.9, label=f"L{j}")
            ax_r.plot(joint_cmd_t, joint_cmd[:, 7 + j], linewidth=0.9, label=f"R{j}")
        ax_l.legend(fontsize=6, ncol=4, loc="upper right")
        ax_r.legend(fontsize=6, ncol=4, loc="upper right")
    else:
        for ax in (ax_l, ax_r):
            ax.text(0.5, 0.5, "no joint_cmd recorded", ha="center", va="center", transform=ax.transAxes)
    for ax, tag in ((ax_l, "left arm"), (ax_r, "right arm")):
        ax.set_xlabel("t [s]")
        ax.set_ylabel("joint cmd [rad]")
        ax.set_title(f"Commanded joint positions - {tag}")
        ax.grid(alpha=0.3)

    fig.suptitle(_title_from_sidecar(trace_path), fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--trace", type=Path, required=True, help="NPZ trace file saved by trajectory.py/ik_controller.py")
    parser.add_argument("--output", type=Path, default=None, help="output PNG path (default: <trace>.png)")
    args = parser.parse_args()
    trace = args.trace.expanduser().resolve()
    if not trace.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace}")
    out = plot_trace(trace, args.output.expanduser().resolve() if args.output else None)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
