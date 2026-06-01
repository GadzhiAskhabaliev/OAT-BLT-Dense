#!/usr/bin/env bash
# Paired SR eval for dense checkpoint ladder (300 / 500 / 700).
#
# Usage:
#   PHASE=A bash scripts/cluster/run_ladder_sr_eval.sh
#   BEST=ep-0500 PHASE=B bash scripts/cluster/run_ladder_sr_eval.sh
#
# Requires: train stopped, checkpoints present (ep-0700 from watcher, not latest.ckpt).

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR/checkpoints}"
PHASE="${PHASE:-A}"
DEVICE="${DEVICE:-cuda:0}"
TEST_START_SEED="${TEST_START_SEED:-1000}"
BEST="${BEST:-}"

# Phase A: ~30 episodes per task (300 total on LIBERO-10)
SCREEN_N_PER_TASK="${SCREEN_N_PER_TASK:-30}"
# Phase B: 50 episodes per task (500 total; matches default train eval config)
CONFIRM_N_PER_TASK="${CONFIRM_N_PER_TASK:-50}"
CONFIRM_NUM_EXP="${CONFIRM_NUM_EXP:-3}"

HF_REPO_300="${HF_REPO_300:-hackhackhack66666/OAT-BLT-LIBERO-300}"
HF_REPO_500="${HF_REPO_500:-hackhackhack66666/OAT-BLT-LIBERO-500}"
HF_REPO_700="${HF_REPO_700:-hackhackhack66666/OAT-BLT-Libero-700}"

cd "$REPO_HOST"

hf_repo_for_tag() {
  case "$1" in
    ep-0300) echo "$HF_REPO_300" ;;
    ep-0500) echo "$HF_REPO_500" ;;
    ep-0700) echo "$HF_REPO_700" ;;
    *) echo "" ;;
  esac
}

require_ckpt() {
  local tag="$1"
  local hf_repo="${2:-$(hf_repo_for_tag "$1")}"
  local path="$CKPT_DIR/${tag}.ckpt"
  if [[ -f "$path" ]]; then
    echo "$path"
    return 0
  fi
  echo "ERROR: missing $path" >&2
  if [[ "$tag" == "ep-0700" ]]; then
    echo "  Wait for HF watcher (epoch>700) — do NOT substitute latest.ckpt." >&2
    echo "  Or: huggingface-cli download $hf_repo ${tag}.ckpt --local-dir $CKPT_DIR" >&2
  else
    echo "  Or: huggingface-cli download $hf_repo ${tag}.ckpt --local-dir $CKPT_DIR" >&2
  fi
  return 1
}

summarize() {
  local root="$1"
  python3 -c "
import json, glob, os
root = '$root'
rows = []
for p in sorted(glob.glob(os.path.join(root, '*/eval_log.json'))):
    j = json.load(open(p))
    name = p.split('/')[-2]
    sr = j.get('mean_success_rate_mean', j.get('mean_success_rate'))
    stderr = j.get('mean_success_rate_stderr')
    n = j.get('n_test', '?')
    s = f'{name}  SR={sr:.4f}' if sr is not None else f'{name}  SR=?'
    if stderr is not None:
        s += f'  +/- {stderr:.4f}'
    s += f'  (n_test={n})'
    rows.append((sr if sr is not None else -1, s))
for _, line in sorted(rows, key=lambda x: -x[0]):
    print(line)
"
}

run_one() {
  local ckpt="$1"
  local out="$2"
  local n_per_task="$3"
  local num_exp="$4"
  set -o pipefail
  python scripts/eval_policy_sim.py \
    -c "$ckpt" \
    -o "$out" \
    -d "$DEVICE" \
    -n "$num_exp" \
    --overwrite \
    --n-test-per-task "$n_per_task" \
    --test-start-seed "$TEST_START_SEED"
}

if [[ "$PHASE" == "A" ]]; then
  OUT="output/eval/ladder_screen_pt${SCREEN_N_PER_TASK}"
  mkdir -p "$OUT"
  C300=$(require_ckpt ep-0300 "$HF_REPO_300")
  C500=$(require_ckpt ep-0500 "$HF_REPO_500")
  C700=$(require_ckpt ep-0700 "$HF_REPO_700")
  for pair in "ep-0300:$C300" "ep-0500:$C500" "ep-0700:$C700"; do
    tag="${pair%%:*}"
    path="${pair#*:}"
    echo "=== Phase A: $tag (n_test_per_task=$SCREEN_N_PER_TASK) ==="
    run_one "$path" "$OUT/$tag" "$SCREEN_N_PER_TASK" 1
  done
  echo "=== Phase A summary ==="
  summarize "$OUT"
  echo "Set BEST=ep-XXXX and PHASE=B for confirm."

elif [[ "$PHASE" == "B" ]]; then
  if [[ -z "$BEST" ]]; then
    echo "ERROR: set BEST=ep-0300|ep-0500|ep-0700" >&2
    exit 1
  fi
  OUT="output/eval/ladder_confirm_pt${CONFIRM_N_PER_TASK}"
  CKPT=$(require_ckpt "$BEST")
  echo "=== Phase B: $BEST (n_test_per_task=$CONFIRM_N_PER_TASK, num_exp=$CONFIRM_NUM_EXP) ==="
  run_one "$CKPT" "$OUT/$BEST" "$CONFIRM_N_PER_TASK" "$CONFIRM_NUM_EXP"
  echo "=== Phase B summary ==="
  summarize "$OUT"
  echo "Compare mean SR to OAT8 paper reference 56.3% (LIBERO-10, external)."

else
  echo "ERROR: PHASE must be A or B" >&2
  exit 1
fi
