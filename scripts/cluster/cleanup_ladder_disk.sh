#!/usr/bin/env bash
# Optional disk cleanup on cluster — run ONLY when user confirms eval is done.
#
# Usage (dry-run):
#   DRY_RUN=1 bash scripts/cluster/cleanup_ladder_disk.sh
# Apply:
#   bash scripts/cluster/cleanup_ladder_disk.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
REPO="${REPO:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

echo "=== Cleanup candidates (keeps ep-0300/0500/0700.ckpt on HF; local optional) ==="

# Partial / failed eval artifacts
run rm -rf "${REPO}/${RUN_DIR%/}/../eval/ladder_screen_pt30"
run rm -rf "${REPO}/output/eval/ladder_screen_pt30"

# HF git-xet staging (if present)
run rm -rf "${HOME}/hf_push"

# Duplicate local ckpt if already on HF (saves ~2.2GB) — uncomment when sure:
# run rm -f "${REPO}/${RUN_DIR}/checkpoints/ep-0300.ckpt"
# run rm -f "${REPO}/${RUN_DIR}/checkpoints/ep-0500.ckpt"
# run rm -f "${REPO}/${RUN_DIR}/checkpoints/ep-0700.ckpt"

# latest.ckpt only if train is stopped and not resuming:
# run rm -f "${REPO}/${RUN_DIR}/checkpoints/latest.ckpt"

echo "Done. Set DRY_RUN=0 to apply."
