#!/usr/bin/env bash
# Launch counterfactual early-stop watcher in tmux.
# This is CPU-only and DOES NOT stop training.

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
SESSION="${SESSION:-oat_early_stop_watch}"
INTERVAL_SEC="${INTERVAL_SEC:-3600}"
MIN_EPOCH="${MIN_EPOCH:-0}"

cd "${REPO_HOST}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

mkdir -p "${RUN_DIR}"
chmod +x scripts/watch_early_stop_report.py

tmux new-session -d -s "${SESSION}" \
  "cd ${REPO_HOST} && python -u scripts/watch_early_stop_report.py \
    --run-dir ${RUN_DIR} \
    --min-epoch ${MIN_EPOCH} \
    --interval-sec ${INTERVAL_SEC} \
    2>&1 | tee ${RUN_DIR}/early_stop_watch.log"

echo "Started tmux watcher: ${SESSION}"
echo "  run_dir=${RUN_DIR}"
echo "  interval_sec=${INTERVAL_SEC}, min_epoch=${MIN_EPOCH}"
echo "  mode=counterfactual_only (no real stop)"
echo "Attach: tmux attach -t ${SESSION}"
