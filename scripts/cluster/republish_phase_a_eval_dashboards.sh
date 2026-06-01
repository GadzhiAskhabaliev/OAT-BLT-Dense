#!/usr/bin/env bash
# Regenerate + re-upload fixed Phase A sim eval dashboards (per-task bars were empty).
set -euo pipefail

if [[ -z "${REPO_HOST:-}" ]]; then
  if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
    REPO_HOST="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
  else
    REPO_HOST="${HOME}/OAT-RoboMimic-Fine-tune/BLT-OAT"
  fi
fi
ROOT="${ROOT:-output/eval/ladder_screen_pt30}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN" >&2
  exit 1
fi

cd "$REPO_HOST"

republish() {
  local tag="$1"
  local repo="$2"
  local out="$ROOT/$tag"
  local run_log="$ROOT/${tag}_run.log"
  echo "=== fix dashboard $tag -> $repo ==="
  python scripts/publish_ladder_eval_to_hf.py \
    --eval-dir "$out" \
    --tag "$tag" \
    --repo-id "$repo" \
    --run-log "$run_log" \
    --hf-prefix sim_eval \
    --legacy-hf-layout \
    --dashboard-only \
    --phase-label "Phase A screen (30 ep/task)"
}

republish ep-0300 hackhackhack66666/OAT-BLT-LIBERO-300
republish ep-0500 hackhackhack66666/OAT-BLT-LIBERO-500
republish ep-0700 hackhackhack66666/OAT-BLT-Libero-700

echo "ALL dashboards fixed"
