#!/usr/bin/env bash
# Terminal 2 of the TWIST2 trajectory-replay workflow: starts the
# trajectory-replay driver that publishes 35D mimic_obs to Redis for
# the TWIST2 sim server to consume, and logs target-vs-actual
# wrist_yaw_link tracking metrics (torso_link frame).
#
# Prerequisite: activate a conda env that has
#   pinocchio + scipy + redis + numpy
# BEFORE running this script. Example:
#
#   conda activate env_isaacgym
#   bash sim2sim_traj_replay.sh
#
# Run this AFTER ``bash sim2sim_traj.sh`` (Terminal 1) is already up.
#
# Extra flags are forwarded verbatim to trajectory_replay_sim2sim.py
# via "$@", e.g.:
#
#   bash sim2sim_traj_replay.sh --ik_max_iters 300 --disable_plot

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

TRAJECTORY_CSV=${TRAJECTORY_CSV:-${SCRIPT_DIR}/assets/trajectories/traj1.csv}
# Must match the sim server's --policy_frequency (50 Hz by default in
# sim2sim_traj.sh -- see the comment there for why).
CONTROL_FREQUENCY=${CONTROL_FREQUENCY:-50}
RAMP_IN_SEC=${RAMP_IN_SEC:-2.0}
REDIS_IP=${REDIS_IP:-localhost}

cd "${SCRIPT_DIR}/deploy_real"

python trajectory_replay_sim2sim.py \
    --trajectory_csv "${TRAJECTORY_CSV}" \
    --redis_ip "${REDIS_IP}" \
    --control_frequency "${CONTROL_FREQUENCY}" \
    --ramp_in_sec "${RAMP_IN_SEC}" \
    "$@"
