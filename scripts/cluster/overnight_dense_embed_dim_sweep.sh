#!/usr/bin/env bash
# Sequential embed-dim sweep on a dedicated GPU (default: 1).
# Writes RUN_MANIFEST lines for post-hoc summarize/plot scripts.
#
# Usage:
#   CUDA_DEVICE=1 USE_TASK_UID=true bash scripts/cluster/overnight_dense_embed_dim_sweep.sh
#
# policy.embed_dim MUST equal policy.dense_feature_dim.

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
CONTAINER="${CONTAINER:-oat_robomimic_askhabaliev_gs}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-/workspace/OAT-RoboMimic-Fine-tune/checkpoints/hf_oat/tokenizer_ep-0950_mse-0.002.ckpt}"
USE_TASK_UID="${USE_TASK_UID:-true}"
DIMS="${DIMS:-128 384 512}"
MANIFEST="${REPO_HOST}/output/long/RUN_MANIFEST.md"

batch_for_dim() {
  case "$1" in
    512) echo 128 ;;
    384) echo 192 ;;
    *) echo 256 ;;
  esac
}

uid_suffix() {
  if [[ "${USE_TASK_UID}" == "true" ]]; then
    echo "with_uid"
  else
    echo "no_uid"
  fi
}

append_manifest() {
  local line="$1"
  {
    echo ""
    echo "- ${line}"
  } >> "${MANIFEST}"
}

init_manifest() {
  mkdir -p "${REPO_HOST}/output/long"
  if [[ ! -f "${MANIFEST}" ]]; then
    cat > "${MANIFEST}" <<EOF
# Long-run manifest (auto-append)

| Field | Value |
|-------|-------|
| Updated | $(date -Iseconds) |

Each run has \`logs.json\`, \`.hydra/\`, \`checkpoints/\` under \`output/long/<run_name>/\`.

Post-hoc:
\`\`\`bash
python scripts/summarize_training_runs.py --root output/long
python scripts/plot_training_runs.py output/long/<run_a> output/long/<run_b> --out output/long/summary/plots
python scripts/eval_policy_sim.py -c output/long/<run>/checkpoints/latest.ckpt -o eval/<run>
\`\`\`

## Runs
EOF
  fi
}

run_one() {
  local dim="$1"
  local bs uid_tag run_name log_host run_dir
  bs="$(batch_for_dim "$dim")"
  uid_tag="$(uid_suffix)"
  run_name="dense_emb${dim}_${uid_tag}_$(date +%m%d_%H%M%S)"
  log_host="${REPO_HOST}/output/long/${run_name}.log"
  run_dir="output/long/${run_name}"

  echo "========================================"
  echo "GPU=${CUDA_DEVICE} dim=${dim} bs=${bs} uid=${USE_TASK_UID} -> ${run_name}"
  echo "========================================"

  append_manifest "**${run_name}** — embed_dim=${dim}, task_uid=${USE_TASK_UID}, gpu=${CUDA_DEVICE}, started $(date -Iseconds)"

  docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" -i "${CONTAINER}" bash -lc "
    set -euo pipefail
    cd /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT
    export HYDRA_FULL_ERROR=1
    export MUJOCO_GL=egl
    python -u scripts/run_workspace.py --config-name=train_oatpolicy \
      policy.action_tokenizer.checkpoint=${TOKENIZER_CKPT} \
      policy.use_dense_visual_memory=true \
      policy.use_cross_attn=true \
      policy.use_state_memory_tokens=true \
      policy.use_task_uid_in_state_tokens=${USE_TASK_UID} \
      policy.embed_dim=${dim} \
      policy.dense_feature_dim=${dim} \
      policy.n_heads=4 \
      training.resume=false \
      logging.mode=disabled \
      dataloader.batch_size=${bs} \
      val_dataloader.batch_size=${bs} \
      hydra.run.dir=${run_dir}
  " 2>&1 | tee "${log_host}"

  append_manifest "**${run_name}** — finished $(date -Iseconds); logs: \`${run_dir}/logs.json\`"
  echo "Finished dim=${dim} (${run_name})"
}

init_manifest
cd "${REPO_HOST}"
for dim in ${DIMS}; do
  run_one "${dim}"
done

echo "Sweep complete: DIMS=${DIMS} USE_TASK_UID=${USE_TASK_UID}"
