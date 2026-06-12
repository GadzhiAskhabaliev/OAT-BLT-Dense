# Published result mirrors

Snapshots from Hugging Face repos (Phase A/B eval + training dashboards).  
Source of truth remains on HF; these files support the README and offline review.

| Directory | HF repo | Description |
|-----------|---------|-------------|
| `ladder_300/` | [OAT-BLT-LIBERO-300](https://huggingface.co/hackhackhack66666/OAT-BLT-LIBERO-300) | ep-0300 screen eval + training dashboard |
| `ladder_500/` | [OAT-BLT-LIBERO-500](https://huggingface.co/hackhackhack66666/OAT-BLT-LIBERO-500) | ep-0500 screen eval + training dashboard |
| `ladder_700/` | [OAT-BLT-Libero-700](https://huggingface.co/hackhackhack66666/OAT-BLT-Libero-700) | ep-0700 screen eval + training dashboard |
| `phase_b_confirm/` | [sim_eval_phase_b/](https://huggingface.co/hackhackhack66666/OAT-BLT-Libero-700/tree/main/sim_eval_phase_b) | 50 ep/task × 3 seeds |
| `ladder_950/` | cluster `ep-0950_sr-0.527.ckpt` | **best SR** (52.67%), resume in-loop eval + training dashboard |
| `overfit_watcher/` | [overfit_watcher/](https://huggingface.co/hackhackhack66666/OAT-BLT-Libero-700/tree/main/overfit_watcher) | counterfactual early-stop report |

Refresh from HF:

```bash
REPO=hackhackhack66666/OAT-BLT-Libero-700
BASE=https://huggingface.co/$REPO/resolve/main
curl -sL "$BASE/sim_eval/sim_eval_dashboard.png" -o ladder_700/sim_eval_dashboard.png
```
