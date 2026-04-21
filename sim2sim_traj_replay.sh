#!/usr/bin/env bash
# Companion to sim2sim_traj.sh: starts the trajectory-replay driver that
# publishes 35D mimic_obs to Redis for the TWIST2 sim server to consume,
# and logs target-vs-actual wrist_yaw_link tracking metrics (in
# torso_link frame) comparable to
# ``rl_ik_solver/.../trajectory.py``.
#
# Run this AFTER sim2sim_traj.sh (or sim2sim.sh) is already up in a
# separate terminal.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

TRAJECTORY_CSV=${TRAJECTORY_CSV:-/home/rail/rail-unitree/rl_ik_solver/trajectory/recorded_trajectories/traj1.csv}
# Must match the sim server's --policy_frequency (50 Hz by default in
# sim2sim_traj.sh -- see the comment there for why).
CONTROL_FREQUENCY=${CONTROL_FREQUENCY:-50}
RAMP_IN_SEC=${RAMP_IN_SEC:-2.0}
REDIS_IP=${REDIS_IP:-localhost}

# The replay side needs pinocchio (for SLSQP IK + FK) and scipy, which
# are in ``env_isaacgym`` on this workstation. Change if your setup
# differs, or pre-activate the env and set REPLAY_ENV="" to skip the
# auto-activation below.
REPLAY_ENV=${REPLAY_ENV:-env_isaacgym}
if [[ -n "${REPLAY_ENV}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${REPLAY_ENV}" ]]; then
        source ~/miniconda3/bin/activate "${REPLAY_ENV}"
    fi
fi

cd "${SCRIPT_DIR}/deploy_real"

python trajectory_replay_sim2sim.py \
    --trajectory_csv "${TRAJECTORY_CSV}" \
    --redis_ip "${REDIS_IP}" \
    --control_frequency "${CONTROL_FREQUENCY}" \
    --ramp_in_sec "${RAMP_IN_SEC}" \
    "$@"
