#!/usr/bin/env bash
# Launch resume-from-ep-0700 training with periodic sim eval in tmux (docker).
#
#   bash scripts/cluster/launch_resume_train_sim_eval_tmux.sh
#
# Optional env:
#   SESSION=oat_resume_ep0700_sim_eval
#   CUDA_DEVICE=1
#   ROLLOUT_EVERY=50
#   TRAIN_N_TEST=300
#   REDUCE_LR=false

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
SESSION="${SESSION:-oat_resume_ep0700_sim_eval}"
CONTAINER="${CONTAINER:-oat_robomimic_askhabaliev_gs}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
TRAIN_N_TEST="${TRAIN_N_TEST:-300}"
TRAIN_N_PARALLEL_ENVS="${TRAIN_N_PARALLEL_ENVS:-4}"
TRAIN_N_TEST_VIS="${TRAIN_N_TEST_VIS:-0}"
TRAIN_MP_CONTEXT="${TRAIN_MP_CONTEXT:-spawn}"
RESUME_BATCH_SIZE="${RESUME_BATCH_SIZE:-128}"
REDUCE_LR="${REDUCE_LR:-false}"
TARGET_EPOCHS="${TARGET_EPOCHS:-1000}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
RESUME_CKPT="${RESUME_CKPT:-ep-0700}"

chmod +x "${REPO_HOST}/scripts/cluster/run_resume_train_sim_eval.sh" 2>/dev/null || true

tmux kill-session -t "${SESSION}" 2>/dev/null || true
docker exec "${CONTAINER}" bash -lc 'pkill -f "run_workspace.py.*train_oatpolicy" || true' 2>/dev/null || true
sleep 1

WRK="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
tmux new-session -d -s "${SESSION}" \
  "docker exec -e MUJOCO_EGL_DEVICE_ID=${CUDA_DEVICE} -e CUDA_DEVICE=${CUDA_DEVICE} -e ROLLOUT_EVERY=${ROLLOUT_EVERY} -e TRAIN_N_TEST=${TRAIN_N_TEST} -e TRAIN_N_PARALLEL_ENVS=${TRAIN_N_PARALLEL_ENVS} -e TRAIN_N_TEST_VIS=${TRAIN_N_TEST_VIS} -e TRAIN_MP_CONTEXT=${TRAIN_MP_CONTEXT} -e RESUME_BATCH_SIZE=${RESUME_BATCH_SIZE} -e REDUCE_LR=${REDUCE_LR} -e TARGET_EPOCHS=${TARGET_EPOCHS} -e RUN_DIR=${RUN_DIR} -e RESUME_CKPT=${RESUME_CKPT} -i ${CONTAINER} bash -lc 'cd ${WRK} && exec bash scripts/cluster/run_resume_train_sim_eval.sh'"

echo "Started tmux: ${SESSION}"
echo "  Resume: ${RESUME_CKPT} -> latest.ckpt in ${RUN_DIR}"
echo "  lazy_eval=false, rollout_every=${ROLLOUT_EVERY}, n_test=${TRAIN_N_TEST}, GPU=${CUDA_DEVICE}"
echo "  Log: ${RUN_DIR}/resume_sim_eval.log"
echo "Attach: tmux attach -t ${SESSION}"
