# SR Eval Plan: Dense Ladder (300 / 500 / 700)

Run: `oat_dense_with_uid_long_0530_220204`  
Goal: choose the best dense checkpoint by `mean_success_rate` (SR), then compare to OAT paper reference (**OAT8 LIBERO-10 = 56.3% ± ~1.0**).

## Decision Rule

1. Phase A: paired SR screen on `ep-0300`, `ep-0500`, `ep-0700`.
2. Pick top checkpoint (SR primary; if tie &lt;2–3 pp, use `val_loss` / per-task SR).
3. Phase B: confirm only the winner (`n_test_per_task=50`, `num_exp=3`).
4. If **SR(700) > SR(500)** on confirm → resume from `ep-0700`; else stop extension.

`val_loss` / `train_loss` are not selection metrics.

---

## Eval Protocol Semantics (read this first)

**Task suite:** `libero10` → 10 subtasks from `libero_10` (`oat/env/libero/factory.py`).

**`n_test` in `LiberoRunner` = total rollouts across all tasks**, not per task:

```text
episodes_per_task ≈ n_test / 10
num_repeats = ceil(n_test / 10)
```

| Setting | Total `n_test` | ~episodes / task |
|---------|----------------|------------------|
| `--n-test 100` | 100 | **10** (too noisy for screen) |
| `--n-test-per-task 30` | 300 | **30** (Phase A default) |
| `--n-test-per-task 50` | 500 | **50** (Phase B; matches train yaml default) |

**Metric:** `mean_success_rate` = mean over all rollout episodes; per-task keys `{task}/mean_success_rate` in `eval_log.json`.

**Seeds:** `test_start_seed` (default **1000**); episode `i` uses seed `1000 + i`.  
For `num_exp > 1`, each repeat uses `test_start_seed + exp_idx * seed_stride` (default stride = `n_test`).

**Do not use `latest.ckpt`** for the ladder. Use watcher snapshots only.

---

## Checkpoints

| Tag | Source |
|-----|--------|
| `ep-0300` | HF `hackhackhack66666/OAT-BLT-LIBERO-300` if missing locally |
| `ep-0500` | local or HF `OAT-BLT-LIBERO-500` |
| `ep-0700` | **Wait for HF watcher** after epoch &gt; 700 — **do not** rename `latest.ckpt` |

If `ep-0700.ckpt` is missing, wait or:

```bash
huggingface-cli download hackhackhack66666/OAT-BLT-Libero-700 ep-0700.ckpt \
  --local-dir "$RUN/checkpoints"
```

---

## Automated Runner

**Full pipeline (eval → dashboard → HF `sim_eval/*` per repo, order 300→500→700):**

```bash
export HF_TOKEN=hf_...
bash scripts/cluster/launch_ladder_eval_pipeline_tmux.sh
# attach: tmux attach -t oat_ladder_pipeline
```

Uses `--overwrite`, `pipefail`, uploads `sim_eval/eval_log.json`, `sim_eval_dashboard.png`, `eval_summary.md`, `eval_run.log`.

**Phase A only (no HF publish):**

```bash
cd ~/OAT-RoboMimic-Fine-tune/BLT-OAT
chmod +x scripts/cluster/run_ladder_sr_eval.sh

# Phase A (~30 ep/task, 300 total per ckpt)
PHASE=A bash scripts/cluster/run_ladder_sr_eval.sh

# Phase B (50 ep/task, 500 total; 3 independent seed blocks)
BEST=ep-0500 PHASE=B bash scripts/cluster/run_ladder_sr_eval.sh
```

Env overrides: `SCREEN_N_PER_TASK`, `CONFIRM_N_PER_TASK`, `TEST_START_SEED`, `DEVICE`, `RUN_DIR`.

---

## Manual Commands

### Phase A (screen)

```bash
cd ~/OAT-RoboMimic-Fine-tune/BLT-OAT
RUN=output/long/oat_dense_with_uid_long_0530_220204
CKPT="$RUN/checkpoints"
OUT=output/eval/ladder_screen_pt30

for tag in ep-0300 ep-0500 ep-0700; do
  python scripts/eval_policy_sim.py \
    -c "$CKPT/${tag}.ckpt" \
    -o "$OUT/${tag}" \
    -d cuda:0 -n 1 \
    --n-test-per-task 30 \
    --test-start-seed 1000
done
```

### Phase B (confirm)

```bash
BEST=ep-0500   # Phase A winner
python scripts/eval_policy_sim.py \
  -c "$CKPT/${BEST}.ckpt" \
  -o output/eval/ladder_confirm_pt50/${BEST} \
  -d cuda:0 -n 3 \
  --n-test-per-task 50 \
  --test-start-seed 1000
```

### Summarize

```bash
python3 -c "
import json, glob, os
for p in sorted(glob.glob('output/eval/ladder_*/*/eval_log.json')):
    j=json.load(open(p))
    sr=j['mean_success_rate_mean']
    se=j.get('mean_success_rate_stderr')
    print(p, f'SR={sr:.3f}', f'±{se:.3f}' if se else '', f'n_test={j[\"n_test\"]}')
"
```

---

## Compare With OAT 56.3% (no local baseline run)

Paper OAT8: **mean SR over 10 LIBERO-10 tasks**, ~50–100 episodes per task, reported **56.3% ± ~1.0**.

| Your confirm SR | Interpretation |
|-----------------|----------------|
| ≥ 58% | above reported OAT8 |
| 54–58% | comparable within typical eval noise |
| &lt; 50% | likely below OAT reference |

**Report caveats:**

- 56.3% is from paper/site, not reproduced on your cluster.
- Your model: dense memory, cross-attn, `task_uid`, `embed_dim=256`.
- Same **metric name**, possibly different sim stack / seeds / episode count.

Optional strict claim: one pooled OAT8 ckpt eval on your machine (Phase B protocol only).

---

## Tie-breaking (Phase A)

If |SR_a − SR_b| &lt; **3 pp**:

1. Prefer lower `val_loss` at that epoch (from `ep-XXXX_upload_meta.json` or `logs.json`).
2. Or run Phase B on **two** top candidates if GPU time allows.

---

## Pre-flight Checklist

- [ ] Train stopped (eval exclusive GPU)
- [ ] `ep-0700.ckpt` from watcher / HF (not `latest.ckpt`)
- [ ] `ep-0300` downloaded if only meta/json left locally
- [ ] Phase A: `--n-test-per-task 30` (not raw `--n-test 100`)
- [ ] Phase B: `--n-test-per-task 50`, `-n 3`, `--test-start-seed 1000`
- [ ] Record `eval_log.json` + per-task SR for report
- [ ] Compare confirm SR to **56.3%** with caveats above

---

## Resume After Eval

If confirm: **SR(700) > SR(500)**:

```bash
cp checkpoints/ep-0700.ckpt checkpoints/latest.ckpt
# resume train with training.resume=True; optional policy_lr x0.1, +200-300 epochs
```
