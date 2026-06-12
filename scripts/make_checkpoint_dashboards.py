#!/usr/bin/env python3
"""
Build training + sim-eval dashboards (same style as ladder ep-0300/0500/0700).

Usage:
  python scripts/make_checkpoint_dashboards.py \\
    --run-dir output/long/oat_dense_with_uid_long_0530_220204 \\
    --epoch 950 \\
    --tag ep-0950_sr-0.527 \\
    --out docs/results/ladder_950
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.publish_ladder_eval_to_hf import make_eval_dashboard, write_eval_summary
from scripts.publish_ladder_hf_readmes import load_jsonl, make_dashboard, snapshot_metrics


def epoch_row(rows: List[Dict[str, Any]], epoch: int) -> Dict[str, Any]:
    for r in reversed(rows):
        if int(r.get("epoch", -1)) == epoch:
            return r
    return {}


def logs_to_eval_log(
    row: Dict[str, Any],
    tag: str,
    checkpoint: str,
    n_test: int = 300,
) -> Dict[str, Any]:
    """Convert in-loop rollout row from logs.json to eval_log.json schema."""
    ev: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "num_exp": 1,
        "n_test": n_test,
        "n_test_per_task": n_test // 10,
        "num_tasks": 10,
        "episodes_per_task_approx": n_test / 10.0,
        "test_start_seed": 1000,
        "seed_stride": n_test,
        "task_suite": "libero10",
        "mean_success_rate_mean": row.get("mean_success_rate"),
    }
    for key, val in row.items():
        if key.endswith("/mean_success_rate"):
            ev[f"{key}_mean"] = val
    return ev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or run_dir.name
    rows = load_jsonl(run_dir / "logs.json")
    row = epoch_row(rows, args.epoch)

    snap = snapshot_metrics(rows, args.epoch)
    if row.get("mean_success_rate") is not None:
        snap["mean_success_rate"] = row["mean_success_rate"]
    if row.get("train_loss") is not None:
        snap["train_loss"] = row["train_loss"]
    if row.get("val_loss") is not None:
        snap["val_loss"] = row["val_loss"]
    if row.get("test_reconst_mse") is not None:
        snap["test_reconst_mse"] = row["test_reconst_mse"]

    ckpt_path = str(run_dir / "checkpoints" / f"{args.tag}.ckpt")
    training_png = out_dir / "training_metrics_dashboard.png"
    make_dashboard(run_dir, args.epoch, training_png, run_name)
    # Re-save with SR annotation via patched snap in PNG is inside make_dashboard;
    # regenerate with SR note by updating snap file for README consumers.
    (out_dir / "training_snapshot.json").write_text(
        json.dumps(snap, indent=2), encoding="utf-8"
    )

    if "mean_success_rate" not in row:
        raise SystemExit(f"No mean_success_rate in logs.json at epoch {args.epoch}")

    ev = logs_to_eval_log(row, args.tag, ckpt_path)
    eval_dir = out_dir
    (eval_dir / "eval_log.json").write_text(json.dumps(ev, indent=2, sort_keys=True), encoding="utf-8")
    make_eval_dashboard(eval_dir, out_dir / "sim_eval_dashboard.png", args.tag)
    write_eval_summary(eval_dir, args.tag, phase_label="in-loop screen (30 ep/task)")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
