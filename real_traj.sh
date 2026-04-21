#!/usr/bin/env bash
# Terminal 1 of the TWIST2 real-robot trajectory-tracking workflow:
# brings up the G1 low-level policy server on the real robot
# (server_low_level_g1_real.py) at the policy's native 50 Hz.
#
# Prerequisite: activate a conda env that has
#   unitree_sdk2py + onnxruntime + torch + redis + numpy
# BEFORE running this script. Example:
#
#   conda activate gmr
#   bash real_traj.sh
#
# Terminal 2: in a separate shell, activate an env that has
# pinocchio + scipy + redis, then run ``bash sim2sim_traj_replay.sh``
# -- the SAME replay driver works for both sim and real because they
# share the Redis state/action schema. Recommend bumping
# ``RAMP_IN_SEC`` to 3.0-5.0 for hardware safety:
#
#   RAMP_IN_SEC=5.0 bash sim2sim_traj_replay.sh
#
# SAFETY:
#  * Bring the robot up on a stand or harness before running.
#  * Verify the first CSV pose is close to a safe posture; ramp-in
#    will drive the arms there regardless.
#  * The trajectory CSV must be in torso_link frame (same convention
#    as rl_ik_solver/trajectory/recorded_trajectories/*.csv).
#  * Hit Ctrl-C on either terminal to stop cleanly; the replay side
#    publishes a final hold pose on exit.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CKPT_PATH=${CKPT_PATH:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}
CONFIG_PATH=${CONFIG_PATH:-robot_control/configs/g1.yaml}
NET_IFACE=${NET_IFACE:-wlp0s20f3}

cd "${SCRIPT_DIR}/deploy_real"

python server_low_level_g1_real.py \
    --policy "${CKPT_PATH}" \
    --config "${CONFIG_PATH}" \
    --device cuda \
    --net "${NET_IFACE}" \
    "$@"
