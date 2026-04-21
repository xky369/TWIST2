#!/usr/bin/env bash
# Replay a recorded bimanual wrist EE trajectory through the TWIST2
# sim2sim stack and report tracking metrics compatible with
# rl_ik_solver/trajectory.py.
#
# Two-terminal workflow:
#   1) This script OR ``bash sim2sim.sh`` first, to bring up the MuJoCo
#      + ONNX policy server (server_low_level_g1_sim.py).
#   2) In another terminal, run ``bash sim2sim_traj_replay.sh`` (or the
#      explicit ``python ...trajectory_replay_sim2sim.py`` below) in an
#      env that has pinocchio + scipy + redis (env_isaacgym works).
#
# This file only launches terminal (1); it is a wrapper around
# sim2sim.sh that keeps the conda env activation explicit so the
# replay side can document which env it needs separately.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CKPT_PATH=${CKPT_PATH:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}
XML_PATH=${XML_PATH:-${SCRIPT_DIR}/assets/g1/g1_sim2sim_29dof.xml}
POLICY_FREQUENCY=${POLICY_FREQUENCY:-100}

# Activate the env that has mujoco + onnxruntime + torch + redis. On
# this workstation that env is ``gmr`` (same one teleop.sh uses).
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "gmr" ]]; then
    source ~/miniconda3/bin/activate gmr
fi

cd "${SCRIPT_DIR}/deploy_real"

python server_low_level_g1_sim.py \
    --xml "${XML_PATH}" \
    --policy "${CKPT_PATH}" \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency "${POLICY_FREQUENCY}" \
    --limit_fps 1
