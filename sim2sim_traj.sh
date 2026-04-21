#!/usr/bin/env bash
# Terminal 1 of the TWIST2 trajectory-replay workflow: brings up the
# MuJoCo + ONNX policy sim server (server_low_level_g1_sim.py) at the
# policy's native 50 Hz frequency.
#
# Prerequisite: activate a conda env that has
#   mujoco + onnxruntime + torch + redis
# BEFORE running this script. Example:
#
#   conda activate gmr
#   bash sim2sim_traj.sh
#
# Terminal 2: in a separate shell, activate an env that has
# pinocchio + scipy + redis, then run ``bash sim2sim_traj_replay.sh``.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CKPT_PATH=${CKPT_PATH:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}
XML_PATH=${XML_PATH:-${SCRIPT_DIR}/assets/g1/g1_sim2sim_29dof.xml}
# 50 Hz matches both the TWIST2 training decimation (humanoid_config.py:
# sim dt=0.005 * decimation=4 => 50 Hz) AND the rl_ik_solver baseline
# control_dt=0.02 (g1_upper_ik.yaml), so the policy runs in-distribution
# and the tracking / smoothness metrics are directly comparable.
POLICY_FREQUENCY=${POLICY_FREQUENCY:-50}

cd "${SCRIPT_DIR}/deploy_real"

python server_low_level_g1_sim.py \
    --xml "${XML_PATH}" \
    --policy "${CKPT_PATH}" \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency "${POLICY_FREQUENCY}" \
    --limit_fps 1
