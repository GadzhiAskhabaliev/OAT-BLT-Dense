#!/usr/bin/env bash
# Phase B confirm eval for ladder winner + HF publish.
#
#   export HF_TOKEN=hf_...
#   export MUJOCO_GL=egl
#   BEST=ep-0700 bash scripts/cluster/run_ladder_phase_b_pipeline.sh

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
BEST="${BEST:-ep-0700}"
DEVICE="${DEVICE:-cuda:0}"
CONFIRM_N_PER_TASK="${CONFIRM_N_PER_TASK:-50}"
CONFIRM_NUM_EXP="${CONFIRM_NUM_EXP:-3}"
TEST_START_SEED="${TEST_START_SEED:-1000}"
OUT_ROOT="${OUT_ROOT:-output/eval/ladder_confirm_pt${CONFIRM_N_PER_TASK}}"
PIPELINE_LOG="${PIPELINE_LOG:-$OUT_ROOT/phase_b.log}"

HF_REPO_700="${HF_REPO_700:-hackhackhack66666/OAT-BLT-Libero-700}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN before Phase B" >&2
  exit 1
fi

cd "$REPO_HOST"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() { echo "[phase-b] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

CKPT="$CKPT_DIR/${BEST}.ckpt"
OUT="$OUT_ROOT/$BEST"
RUN_LOG="$OUT_ROOT/${BEST}_run.log"

if [[ ! -f "$CKPT" ]]; then
  log "ERROR missing $CKPT"
  exit 1
fi

log "=== Phase B: $BEST n_test_per_task=$CONFIRM_N_PER_TASK num_exp=$CONFIRM_NUM_EXP device=$DEVICE ==="
python scripts/eval_policy_sim.py \
  -c "$CKPT" \
  -o "$OUT" \
  -d "$DEVICE" \
  -n "$CONFIRM_NUM_EXP" \
  --overwrite \
  --n-test-per-task "$CONFIRM_N_PER_TASK" \
  --test-start-seed "$TEST_START_SEED" \
  2>&1 | tee "$RUN_LOG"

if [[ ! -f "$OUT/eval_log.json" ]]; then
  log "ERROR missing $OUT/eval_log.json"
  exit 1
fi

SR=$(python3 -c "import json; j=json.load(open('$OUT/eval_log.json')); v=j.get('mean_success_rate_mean', j.get('mean_success_rate')); print(f'{100*v:.2f}%' if v is not None else '?')")
log "eval $BEST done  mean_SR=$SR"

log "=== publish $BEST -> $HF_REPO_700 (sim_eval_phase_b/, no overwrite of sim_eval/) ==="
python scripts/publish_ladder_eval_to_hf.py \
  --eval-dir "$OUT" \
  --tag "$BEST" \
  --repo-id "$HF_REPO_700" \
  --run-log "$RUN_LOG" \
  --hf-prefix sim_eval_phase_b \
  --artifact-label "phase_b_confirm_pt${CONFIRM_N_PER_TASK}_${BEST}" \
  --phase-label "Phase B confirm (50 ep/task, num_exp=3)"

log "PHASE_B ALL_DONE $BEST SR=$SR"
