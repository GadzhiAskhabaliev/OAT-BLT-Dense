#!/usr/bin/env bash
# Launch embed-dim sweep in tmux on GPU 1 (main run should use GPU 0).
set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
SESSION="${SESSION:-oat_dense_embed_sweep_night}"
USE_TASK_UID="${USE_TASK_UID:-true}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

cd "${REPO_HOST}"
chmod +x scripts/cluster/overnight_dense_embed_dim_sweep.sh

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd ${REPO_HOST} && CUDA_DEVICE=${CUDA_DEVICE} USE_TASK_UID=${USE_TASK_UID} bash scripts/cluster/overnight_dense_embed_dim_sweep.sh 2>&1 | tee output/long/overnight_embed_sweep_master.log"

echo "Started tmux: ${SESSION}"
echo "  USE_TASK_UID=${USE_TASK_UID} CUDA_DEVICE=${CUDA_DEVICE}"
echo "Attach: tmux attach -t ${SESSION}"
