#!/usr/bin/env bash
# Poll training logs/checkpoints and run eval_policy_sim.py on rollout milestones.
# Runs OUTSIDE train process (avoids CUDA+fork BrokenPipe in LiberoRunner).
#
#   EVAL_DEVICE=cuda:0 ROLLOUT_EVERY=50 bash scripts/cluster/run_periodic_sim_eval_watcher.sh

set -euo pipefail

if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
  REPO_HOST="${REPO_HOST:-/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT}"
else
  REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
fi

RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
N_TEST_PER_TASK="${N_TEST_PER_TASK:-30}"
POLL_SEC="${POLL_SEC:-120}"
OUT_ROOT="${OUT_ROOT:-output/eval/resume_periodic}"
LOG_FILE="${REPO_HOST}/${RUN_DIR}/periodic_sim_eval_watcher.log"

log() { echo "[sim-watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

max_epoch_from_logs() {
  local logs="${REPO_HOST}/${RUN_DIR}/logs.json"
  [[ -f "$logs" ]] || { echo 0; return; }
  python3 - "$logs" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
epochs = []
for line in p.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "epoch" in row:
        epochs.append(int(row["epoch"]))
print(max(epochs) if epochs else 0)
PY
}

latest_ckpt_epoch() {
  local ckpt_dir="${REPO_HOST}/${RUN_DIR}/checkpoints"
  python3 - "$ckpt_dir" <<'PY'
import re, sys
from pathlib import Path
ckpt_dir = Path(sys.argv[1])
best = 0
for p in ckpt_dir.glob("ep-*.ckpt"):
    m = re.match(r"ep-(\d+)", p.stem)
    if m:
        best = max(best, int(m.group(1)))
print(best)
PY
}

run_eval() {
  local milestone="$1"
  local ckpt="${REPO_HOST}/${RUN_DIR}/checkpoints/latest.ckpt"
  local out="${REPO_HOST}/${OUT_ROOT}/ep-$(printf '%04d' "$milestone")"
  if [[ ! -f "$ckpt" ]]; then
    log "skip milestone=${milestone}: no latest.ckpt"
    return 1
  fi
  log "eval milestone=${milestone} ckpt=latest.ckpt n_per_task=${N_TEST_PER_TASK} device=${EVAL_DEVICE}"
  cd "${REPO_HOST}"
  export MUJOCO_GL=egl
  python -u scripts/eval_policy_sim.py \
    --checkpoint "$ckpt" \
    --output_dir "$out" \
    --device "$EVAL_DEVICE" \
    --n-test-per-task "$N_TEST_PER_TASK" \
    --overwrite
}

cd "${REPO_HOST}"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

log "watcher start run_dir=${RUN_DIR} every=${ROLLOUT_EVERY} ep poll=${POLL_SEC}s"
last_eval_epoch=-1

while true; do
  train_epoch="$(max_epoch_from_logs)"
  ckpt_epoch="$(latest_ckpt_epoch)"
  progress_epoch="$train_epoch"
  if (( ckpt_epoch > progress_epoch )); then
    progress_epoch="$ckpt_epoch"
  fi

  # next milestone strictly after last eval (700 -> 750 -> 800 ...)
  next_milestone=$(( (last_eval_epoch / ROLLOUT_EVERY + 1) * ROLLOUT_EVERY ))
  if (( last_eval_epoch < 0 )); then
    next_milestone=$ROLLOUT_EVERY
    while (( next_milestone <= progress_epoch )); do
      next_milestone=$(( next_milestone + ROLLOUT_EVERY ))
    done
    if (( progress_epoch >= ROLLOUT_EVERY )); then
      next_milestone=$progress_epoch
      next_milestone=$(( (next_milestone / ROLLOUT_EVERY) * ROLLOUT_EVERY ))
      if (( next_milestone <= last_eval_epoch )); then
        next_milestone=$(( next_milestone + ROLLOUT_EVERY ))
      fi
    fi
  fi

  if (( progress_epoch >= next_milestone && next_milestone > last_eval_epoch )); then
    if run_eval "$next_milestone"; then
      last_eval_epoch="$next_milestone"
      log "done milestone=${next_milestone}; train_epoch=${train_epoch} ckpt_epoch=${ckpt_epoch}"
    else
      log "eval failed milestone=${next_milestone}; retry later"
    fi
  else
    log "poll train_epoch=${train_epoch} ckpt_epoch=${ckpt_epoch} next_milestone=${next_milestone} last_eval=${last_eval_epoch}"
  fi
  sleep "$POLL_SEC"
done
