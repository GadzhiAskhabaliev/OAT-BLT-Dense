#!/usr/bin/env bash
# Launch Phase B confirm eval in tmux (docker).
#
#   export HF_TOKEN=hf_...
#   BEST=ep-0700 bash scripts/cluster/launch_ladder_phase_b_tmux.sh

set -euo pipefail

REPO_HOST="${REPO_HOST:-$HOME/OAT-RoboMimic-Fine-tune/BLT-OAT}"
SESSION="${SESSION:-oat_ladder_phase_b}"
CONTAINER="${CONTAINER:-oat_robomimic_askhabaliev_gs}"
DEVICE="${DEVICE:-cuda:0}"
BEST="${BEST:-ep-0700}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN=hf_..." >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
docker exec "${CONTAINER}" bash -lc 'pkill -f eval_policy_sim.py || true' 2>/dev/null || true
sleep 1

chmod +x "${REPO_HOST}/scripts/cluster/run_ladder_phase_b_pipeline.sh" 2>/dev/null || true

WRK="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
tmux new-session -d -s "${SESSION}" \
  "docker exec -i ${CONTAINER} bash -lc 'cd ${WRK} && export REPO_HOST=${WRK} MUJOCO_GL=egl HF_TOKEN=${HF_TOKEN} DEVICE=${DEVICE} BEST=${BEST} && exec bash scripts/cluster/run_ladder_phase_b_pipeline.sh 2>&1 | tee -a output/eval/ladder_confirm_pt50/phase_b.log'"

echo "Started tmux: ${SESSION}"
echo "  Phase B: ${BEST} (50 ep/task, num_exp=3) -> HF sim_eval_phase_b/ (sim_eval/ untouched)"
echo "Attach: tmux attach -t ${SESSION}"
