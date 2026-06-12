#!/usr/bin/env bash
# Phase B confirm eval for ep-0950_sr-0.527 (no HF publish).

set -euo pipefail

if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
  REPO_HOST="${REPO_HOST:-/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT}"
else
  REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
fi

RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
CKPT_TAG="${CKPT_TAG:-ep-0950_sr-0.527}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
N_PARALLEL_ENVS="${N_PARALLEL_ENVS:-4}"
CONFIRM_N_PER_TASK="${CONFIRM_N_PER_TASK:-50}"
CONFIRM_NUM_EXP="${CONFIRM_NUM_EXP:-3}"
TEST_START_SEED="${TEST_START_SEED:-1000}"

CKPT="${REPO_HOST}/${RUN_DIR}/checkpoints/${CKPT_TAG}.ckpt"
OUT="${REPO_HOST}/output/eval/ladder_confirm_pt${CONFIRM_N_PER_TASK}/${CKPT_TAG}"
LOG="${REPO_HOST}/${RUN_DIR}/confirm_ep0950_eval.log"

log() { echo "[confirm-ep0950] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f "${CKPT}" ]]; then
  log "ERROR: missing ${CKPT}" >&2
  exit 1
fi

cd "${REPO_HOST}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
# EGL index is relative to visible devices (always 0 when one GPU is exposed).
export MUJOCO_EGL_DEVICE_ID=0

exec > >(tee -a "${LOG}") 2>&1

log "CKPT=${CKPT_TAG} n_test_per_task=${CONFIRM_N_PER_TASK} num_exp=${CONFIRM_NUM_EXP} n_parallel_envs=${N_PARALLEL_ENVS} device=${DEVICE}"

python -u scripts/eval_policy_sim.py \
  -c "${CKPT}" \
  -o "${OUT}" \
  -d "${DEVICE}" \
  -n "${CONFIRM_NUM_EXP}" \
  --overwrite \
  --n-test-per-task "${CONFIRM_N_PER_TASK}" \
  --test-start-seed "${TEST_START_SEED}" \
  --n-parallel-envs "${N_PARALLEL_ENVS}" \
  --mp-context spawn

python3 -c "
import json
j=json.load(open('${OUT}/eval_log.json'))
sr=j.get('mean_success_rate_mean', j.get('mean_success_rate'))
se=j.get('mean_success_rate_stderr')
line=f'mean_SR={100*sr:.2f}%' if sr is not None else 'mean_SR=?'
if se is not None:
    line += f' +/- {100*se:.2f}%'
print('[confirm-ep0950] DONE', line)
"
