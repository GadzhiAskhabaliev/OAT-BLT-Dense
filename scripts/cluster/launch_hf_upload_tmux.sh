#!/usr/bin/env bash
# Launch HF checkpoint upload watcher for epoch snapshots (train continues).
#
# Checkpoint ladder (SR eval): 300, 500, 700, 900, 1100 — separate HF repos.
#
# Example (ep 700 only):
#   export HF_TOKEN=hf_...
#   TARGET_EPOCHS="700" \
#   EPOCH_REPOS="700=hackhackhack66666/OAT-BLT-Libero-700" \
#   bash scripts/cluster/launch_hf_upload_tmux.sh
#
# Example (remaining ladder):
#   TARGET_EPOCHS="700 900 1100" \
#   EPOCH_REPOS="700=hackhackhack66666/OAT-BLT-Libero-700 900=hackhackhack66666/OAT-BLT-Libero-900 1100=hackhackhack66666/OAT-BLT-Libero-1100" \
#   bash scripts/cluster/launch_hf_upload_tmux.sh

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
SESSION="${SESSION:-oat_hf_upload_watch}"
TARGET_EPOCHS="${TARGET_EPOCHS:-700}"
HF_REPO="${HF_REPO:-hackhackhack66666/OAT-BLT-LIBERO-300}"
# Space-separated EPOCH=REPO pairs
EPOCH_REPOS="${EPOCH_REPOS:-700=hackhackhack66666/OAT-BLT-Libero-700}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
POST_CHECKPOINT_DELAY_SEC="${POST_CHECKPOINT_DELAY_SEC:-120}"
UPLOAD_METHOD="${UPLOAD_METHOD:-hub}"

cd "${REPO_HOST}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export HF_TOKEN=hf_... before launching."
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

mkdir -p "${RUN_DIR}"
chmod +x scripts/watch_hf_checkpoint_upload.py
python3 -m pip install -q huggingface_hub 2>/dev/null || true
git xet install 2>/dev/null || true

EPOCH_REPO_FLAGS=""
for pair in ${EPOCH_REPOS}; do
  EPOCH_REPO_FLAGS="${EPOCH_REPO_FLAGS} --epoch-repo ${pair}"
done

WATCH_CMD="cd ${REPO_HOST} && export HF_TOKEN=${HF_TOKEN} && exec python3 -u scripts/watch_hf_checkpoint_upload.py \
  --run-dir ${RUN_DIR} \
  --target-epochs ${TARGET_EPOCHS} \
  --hf-repo ${HF_REPO}${EPOCH_REPO_FLAGS} \
  --interval-sec ${INTERVAL_SEC} \
  --post-checkpoint-delay-sec ${POST_CHECKPOINT_DELAY_SEC} \
  --method ${UPLOAD_METHOD} \
  --no-exit-when-all-done \
  2>&1 | tee -a ${RUN_DIR}/hf_upload_watch.log"

tmux new-session -d -s "${SESSION}" "${WATCH_CMD}"

echo "Started tmux HF upload watcher: ${SESSION}"
echo "  run_dir=${RUN_DIR}"
echo "  target_epochs=${TARGET_EPOCHS}"
echo "  epoch_repos=${EPOCH_REPOS}"
echo "  method=${UPLOAD_METHOD}"
echo "  training_not_stopped=True"
echo "Attach: tmux attach -t ${SESSION}"
