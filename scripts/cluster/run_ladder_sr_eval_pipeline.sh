#!/usr/bin/env bash
# Phase A ladder: eval 300 → 500 → 700, then publish dashboard + logs to each HF repo.
#
# Usage (inside docker or host with LIBERO):
#   export HF_TOKEN=hf_...
#   export MUJOCO_GL=egl
#   bash scripts/cluster/run_ladder_sr_eval_pipeline.sh
#
# Env: RUN_DIR, DEVICE, SCREEN_N_PER_TASK, TEST_START_SEED, OUT_ROOT, REPO_HOST

set -euo pipefail
set -o pipefail

if [[ -z "${REPO_HOST:-}" ]]; then
  if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
    REPO_HOST="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
  else
    REPO_HOST="${HOME}/OAT-RoboMimic-Fine-tune/BLT-OAT"
  fi
fi
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR/checkpoints}"
OUT_ROOT="${OUT_ROOT:-output/eval/ladder_screen_pt30}"
DEVICE="${DEVICE:-cuda:0}"
SCREEN_N_PER_TASK="${SCREEN_N_PER_TASK:-30}"
TEST_START_SEED="${TEST_START_SEED:-1000}"
PIPELINE_LOG="${PIPELINE_LOG:-$OUT_ROOT/pipeline.log}"

HF_REPO_300="${HF_REPO_300:-hackhackhack66666/OAT-BLT-LIBERO-300}"
HF_REPO_500="${HF_REPO_500:-hackhackhack66666/OAT-BLT-LIBERO-500}"
HF_REPO_700="${HF_REPO_700:-hackhackhack66666/OAT-BLT-Libero-700}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN before running pipeline" >&2
  exit 1
fi

cd "$REPO_HOST"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() { echo "[pipeline] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

hf_repo_for_tag() {
  case "$1" in
    ep-0300) echo "$HF_REPO_300" ;;
    ep-0500) echo "$HF_REPO_500" ;;
    ep-0700) echo "$HF_REPO_700" ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}

require_ckpt() {
  local tag="$1"
  local path="$CKPT_DIR/${tag}.ckpt"
  if [[ -f "$path" ]]; then
    echo "$path"
    return 0
  fi
  log "ERROR missing $path"
  return 1
}

run_eval_one() {
  local tag="$1"
  local ckpt="$2"
  local out="$OUT_ROOT/$tag"
  local run_log="$OUT_ROOT/${tag}_run.log"

  log "=== eval $tag (n_test_per_task=$SCREEN_N_PER_TASK device=$DEVICE) ==="
  python scripts/eval_policy_sim.py \
    -c "$ckpt" \
    -o "$out" \
    -d "$DEVICE" \
    -n 1 \
    --overwrite \
    --n-test-per-task "$SCREEN_N_PER_TASK" \
    --test-start-seed "$TEST_START_SEED" \
    2>&1 | tee "$run_log"

  if [[ ! -f "$out/eval_log.json" ]]; then
    log "ERROR: $out/eval_log.json missing after eval $tag"
    return 1
  fi
  log "eval $tag done"
}

publish_one() {
  local tag="$1"
  local out="$OUT_ROOT/$tag"
  local repo
  repo="$(hf_repo_for_tag "$tag")"
  local run_log="$OUT_ROOT/${tag}_run.log"

  log "=== publish $tag -> $repo ==="
  python scripts/publish_ladder_eval_to_hf.py \
    --eval-dir "$out" \
    --tag "$tag" \
    --repo-id "$repo" \
    --run-log "$run_log" \
    --hf-prefix sim_eval \
    --legacy-hf-layout \
    --phase-label "Phase A screen (${SCREEN_N_PER_TASK} ep/task)"
  log "published $tag"
}

summarize_all() {
  log "=== ladder summary ==="
  python3 -c "
import json, glob, os
root = '$OUT_ROOT'
rows = []
for p in sorted(glob.glob(os.path.join(root, 'ep-*/eval_log.json'))):
    j = json.load(open(p))
    name = p.split('/')[-2]
    sr = j.get('mean_success_rate_mean', j.get('mean_success_rate'))
    n = j.get('n_test', '?')
    line = f'{name}  SR={sr:.4f}  (n_test={n})' if sr is not None else f'{name}  SR=?'
    rows.append((sr if sr is not None else -1, line))
for _, line in sorted(rows, key=lambda x: -x[0]):
    print(line)
"
}

FAILED=0
for tag in ep-0300 ep-0500 ep-0700; do
  if ! ckpt="$(require_ckpt "$tag")"; then
    FAILED=1
    continue
  fi
  if ! run_eval_one "$tag" "$ckpt"; then
    log "FAILED eval $tag"
    FAILED=1
    continue
  fi
  if ! publish_one "$tag"; then
    log "FAILED publish $tag"
    FAILED=1
    continue
  fi
done

summarize_all

if [[ "$FAILED" -ne 0 ]]; then
  log "PIPELINE finished with errors"
  exit 1
fi
log "PIPELINE ALL_DONE"
