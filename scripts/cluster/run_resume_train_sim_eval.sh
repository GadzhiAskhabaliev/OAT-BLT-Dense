#!/usr/bin/env bash
# Resume long-run training from ep-0700 with periodic in-loop LIBERO sim eval.
#
# Prerequisites:
#   - Train/eval jobs stopped; ep-0700.ckpt present under RUN_DIR/checkpoints/
#   - GPU with enough VRAM (default cuda:1)
#
# Usage (host or inside docker):
#   bash scripts/cluster/run_resume_train_sim_eval.sh
#
# Env:
#   RUN_DIR          — existing Hydra run (default: oat_dense_with_uid_long_0530_220204)
#   RESUME_CKPT      — checkpoint tag to seed latest.ckpt (default: ep-0700)
#   CUDA_DEVICE      — visible GPU index (default: 1)
#   ROLLOUT_EVERY    — sim eval every N epochs (default: 50)
#   TRAIN_N_TEST     — total rollouts per eval (default: 300 ≈ 30/task on libero10)
#   REDUCE_LR        — if true, policy_lr/obs_enc_lr ×0.1 (default: false)
#   TARGET_EPOCHS    — stop after this epoch (default: 1000; config allows up to 5001)
#   LOGGING_MODE     — wandb mode (default: disabled)

set -euo pipefail

if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
  REPO_HOST="${REPO_HOST:-/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT}"
else
  REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
fi

RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
RESUME_CKPT="${RESUME_CKPT:-ep-0700}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
TRAIN_N_TEST="${TRAIN_N_TEST:-300}"
TRAIN_N_PARALLEL_ENVS="${TRAIN_N_PARALLEL_ENVS:-4}"
TRAIN_N_TEST_VIS="${TRAIN_N_TEST_VIS:-0}"
TRAIN_MP_CONTEXT="${TRAIN_MP_CONTEXT:-spawn}"
RESUME_BATCH_SIZE="${RESUME_BATCH_SIZE:-128}"
REDUCE_LR="${REDUCE_LR:-false}"
TARGET_EPOCHS="${TARGET_EPOCHS:-1000}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-/workspace/OAT-RoboMimic-Fine-tune/checkpoints/hf_oat/tokenizer_ep-0950_mse-0.002.ckpt}"

CKPT_DIR="${REPO_HOST}/${RUN_DIR}/checkpoints"
SRC_CKPT="${CKPT_DIR}/${RESUME_CKPT}.ckpt"
LATEST_CKPT="${CKPT_DIR}/latest.ckpt"
LOG_HOST="${REPO_HOST}/${RUN_DIR}/resume_sim_eval.log"

log() { echo "[resume-train] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f "${SRC_CKPT}" ]]; then
  log "ERROR: missing ${SRC_CKPT}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"
if [[ "${RESUME_CKPT}" == "latest" ]]; then
  if [[ ! -f "${LATEST_CKPT}" ]]; then
    log "ERROR: RESUME_CKPT=latest but ${LATEST_CKPT} is missing" >&2
    exit 1
  fi
  log "Resuming in-place from existing latest.ckpt (epoch checkpoint already in run dir)"
else
  if [[ -f "${LATEST_CKPT}" ]]; then
    backup="${CKPT_DIR}/latest_before_resume_$(date +%Y%m%d_%H%M%S).ckpt"
    log "Backing up existing latest.ckpt -> $(basename "${backup}")"
    cp -a "${LATEST_CKPT}" "${backup}"
  fi
  log "Seeding latest.ckpt from ${RESUME_CKPT}.ckpt"
  cp -a "${SRC_CKPT}" "${LATEST_CKPT}"
fi

lr_overrides=()
if [[ "${REDUCE_LR}" == "true" ]]; then
  lr_overrides+=(optimizer.policy_lr=5.0e-6 optimizer.obs_enc_lr=1.0e-6)
  log "LR reduced ×0.1 (policy_lr=5e-6, obs_enc_lr=1e-6)"
fi

log "RUN_DIR=${RUN_DIR} GPU=${CUDA_DEVICE} rollout_every=${ROLLOUT_EVERY} n_test=${TRAIN_N_TEST} n_parallel_envs=${TRAIN_N_PARALLEL_ENVS} n_test_vis=${TRAIN_N_TEST_VIS} mp_context=${TRAIN_MP_CONTEXT} target_epochs=${TARGET_EPOCHS}"
log "Baseline Phase B confirm SR: 47.60% ± 1.75% (ep-0700, 50 ep/task, 3 seeds)"
log "Paper ref ~56.3% — compare in-loop SR with same n_test scale when possible"

cd "${REPO_HOST}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
# Keep EGL device aligned with selected CUDA device for robosuite checks.
export MUJOCO_EGL_DEVICE_ID="${CUDA_DEVICE}"

exec > >(tee -a "${LOG_HOST}") 2>&1

python -u scripts/run_workspace.py --config-name=train_oatpolicy \
  policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
  policy.use_dense_visual_memory=true \
  policy.use_cross_attn=true \
  policy.use_state_memory_tokens=true \
  policy.use_task_uid_in_state_tokens=true \
  policy.embed_dim=256 \
  policy.dense_feature_dim=256 \
  task.policy.lazy_eval=false \
  task.policy.env_runner.n_test="${TRAIN_N_TEST}" \
  task.policy.env_runner.n_parallel_envs="${TRAIN_N_PARALLEL_ENVS}" \
  task.policy.env_runner.n_test_vis="${TRAIN_N_TEST_VIS}" \
  +task.policy.env_runner.mp_context="${TRAIN_MP_CONTEXT}" \
  dataloader.batch_size="${RESUME_BATCH_SIZE}" \
  val_dataloader.batch_size="${RESUME_BATCH_SIZE}" \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  val_dataloader.persistent_workers=false \
  dataloader.pin_memory=false \
  val_dataloader.pin_memory=false \
  training.resume=true \
  training.rollout_every="${ROLLOUT_EVERY}" \
  training.num_epochs="${TARGET_EPOCHS}" \
  logging.mode="${LOGGING_MODE}" \
  hydra.run.dir="${RUN_DIR}" \
  "${lr_overrides[@]}"
