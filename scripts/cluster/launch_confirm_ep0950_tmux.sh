#!/usr/bin/env bash
# Phase B confirm eval for best resume checkpoint (ep-0950) in tmux -> docker.
#
#   bash scripts/cluster/launch_confirm_ep0950_tmux.sh
#
# Optional env:
#   SESSION=oat_confirm_ep0950
#   DEVICE=cuda:0
#   CUDA_DEVICE=0
#   N_PARALLEL_ENVS=4

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
SESSION="${SESSION:-oat_confirm_ep0950}"
CONTAINER="${CONTAINER:-oat_robomimic_askhabaliev_gs}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
CKPT_TAG="${CKPT_TAG:-ep-0950_sr-0.527}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
CONFIRM_N_PER_TASK="${CONFIRM_N_PER_TASK:-50}"
CONFIRM_NUM_EXP="${CONFIRM_NUM_EXP:-3}"
TEST_START_SEED="${TEST_START_SEED:-1000}"

chmod +x "${REPO_HOST}/scripts/cluster/run_confirm_ep0950_eval.sh" 2>/dev/null || true

tmux start-server 2>/dev/null || true
tmux kill-session -t "${SESSION}" 2>/dev/null || true
docker exec "${CONTAINER}" bash -lc 'pkill -f eval_policy_sim.py || true' 2>/dev/null || true
sleep 1

WRK="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
# Keep tmux window open even if docker exec fails (attach for debugging).
tmux new-session -d -s "${SESSION}" \
  "docker exec -e MUJOCO_EGL_DEVICE_ID=${CUDA_DEVICE} -e CUDA_DEVICE=${CUDA_DEVICE} -e DEVICE=${DEVICE} -e RUN_DIR=${RUN_DIR} -e CKPT_TAG=${CKPT_TAG} -e N_PARALLEL_ENVS=${N_PARALLEL_ENVS} -e CONFIRM_N_PER_TASK=${CONFIRM_N_PER_TASK} -e CONFIRM_NUM_EXP=${CONFIRM_NUM_EXP} -e TEST_START_SEED=${TEST_START_SEED} -i ${CONTAINER} bash -lc 'cd ${WRK} && bash scripts/cluster/run_confirm_ep0950_eval.sh'; echo; echo exit=\$?; exec bash -l"

echo "Started tmux: ${SESSION}"
echo "  Confirm eval: ${CKPT_TAG} (${CONFIRM_N_PER_TASK} ep/task, num_exp=${CONFIRM_NUM_EXP})"
echo "  n_parallel_envs=${N_PARALLEL_ENVS}, GPU=${CUDA_DEVICE}"
echo "  Log: ${RUN_DIR}/confirm_ep0950_eval.log"
echo "Attach: tmux attach -t ${SESSION}"
