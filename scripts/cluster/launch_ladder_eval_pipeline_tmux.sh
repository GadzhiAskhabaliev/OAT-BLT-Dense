#!/usr/bin/env bash
# Stop broken ladder eval (if any) and start full pipeline in tmux (docker).
#
#   export HF_TOKEN=hf_...
#   bash scripts/cluster/launch_ladder_eval_pipeline_tmux.sh

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
SESSION="${SESSION:-oat_ladder_pipeline}"
CONTAINER="${CONTAINER:-oat_robomimic_askhabaliev_gs}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN=hf_... before launch" >&2
  exit 1
fi

# Stop prior eval (tmux + stray docker exec).
tmux kill-session -t oat_ladder_eval 2>/dev/null || true
tmux kill-session -t "${SESSION}" 2>/dev/null || true
docker exec "${CONTAINER}" bash -lc 'pkill -f eval_policy_sim.py || true' 2>/dev/null || true
sleep 2

chmod +x "${REPO_HOST}/scripts/cluster/run_ladder_sr_eval_pipeline.sh" 2>/dev/null || true

WRK="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
tmux new-session -d -s "${SESSION}" \
  "docker exec -i ${CONTAINER} bash -lc 'cd ${WRK} && export REPO_HOST=${WRK} MUJOCO_GL=egl HF_TOKEN=${HF_TOKEN} DEVICE=${DEVICE} && exec bash scripts/cluster/run_ladder_sr_eval_pipeline.sh 2>&1 | tee -a output/eval/ladder_screen_pt30/pipeline.log'"

echo "Started tmux: ${SESSION}"
echo "  Order: ep-0300 -> ep-0500 -> ep-0700 (eval + HF sim_eval/* per repo)"
echo "Attach: tmux attach -t ${SESSION}"
