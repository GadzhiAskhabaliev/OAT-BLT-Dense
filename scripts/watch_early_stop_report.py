#!/usr/bin/env python3
"""
Counterfactual early-stop watcher for TrainPolicyWorkspace logs.

This script NEVER stops training. It only writes periodic reports:
- <run_dir>/early_stop_report.jsonl
- <run_dir>/early_stop_report.md

It is intended for post-hoc analysis ("where we WOULD have stopped")
without changing the running training job.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalPoint:
    epoch: int
    global_step: Optional[int]
    train_loss: Optional[float]
    val_loss: Optional[float]
    reconst_mse: Optional[float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip incomplete trailing line if training is writing right now.
            continue
    return rows


def extract_eval_points(rows: List[Dict[str, Any]]) -> List[EvalPoint]:
    points: List[EvalPoint] = []
    for r in rows:
        if "val_loss" not in r:
            continue
        points.append(
            EvalPoint(
                epoch=int(r.get("epoch", -1)),
                global_step=r.get("global_step"),
                train_loss=float(r["train_loss"]) if "train_loss" in r else None,
                val_loss=float(r["val_loss"]) if "val_loss" in r else None,
                reconst_mse=(
                    float(r["test_reconst_mse"]) if "test_reconst_mse" in r else None
                ),
            )
        )
    return points


def monotonic_decreasing(values: List[float], tol: float = 1e-9) -> bool:
    return all(values[i + 1] <= values[i] + tol for i in range(len(values) - 1))


def monotonic_increasing(values: List[float], tol: float = 1e-9) -> bool:
    return all(values[i + 1] >= values[i] - tol for i in range(len(values) - 1))


def as_float_list(values: List[Optional[float]]) -> Optional[List[float]]:
    if any(v is None for v in values):
        return None
    return [float(v) for v in values]  # type: ignore[arg-type]


def evaluate_counterfactual(
    rows: List[Dict[str, Any]],
    min_epoch: int,
    val_window: int,
    train_drop_min: float,
    val_rise_min: float,
    reconst_rise_min: float,
    plateau_train_eps: float,
    plateau_val_eps: float,
    plateau_reconst_eps: float,
) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "waiting_logs",
            "reason": "logs.json not available yet",
        }

    last = rows[-1]
    last_epoch = int(last.get("epoch", -1))
    last_step = last.get("global_step")
    last_train = last.get("train_loss")
    if last_epoch < min_epoch:
        return {
            "status": "waiting_min_epoch",
            "reason": f"epoch={last_epoch} < min_epoch={min_epoch}",
            "last_epoch": last_epoch,
            "last_global_step": last_step,
            "last_train_loss": last_train,
        }

    eval_points = extract_eval_points(rows)
    if len(eval_points) < val_window:
        return {
            "status": "waiting_eval_points",
            "reason": (
                f"need at least {val_window} val points, got {len(eval_points)}"
            ),
            "last_epoch": last_epoch,
            "last_global_step": last_step,
            "last_train_loss": last_train,
        }

    recent = eval_points[-val_window:]
    train_vals = as_float_list([p.train_loss for p in recent])
    val_vals = as_float_list([p.val_loss for p in recent])
    rec_vals = as_float_list([p.reconst_mse for p in recent])

    train_drop = None
    val_rise = None
    rec_rise = None
    overfit_val = False
    overfit_reconst = False
    plateau = False

    if train_vals and val_vals:
        train_drop = train_vals[0] - train_vals[-1]
        val_rise = val_vals[-1] - val_vals[0]
        overfit_val = (
            train_drop >= train_drop_min
            and val_rise >= val_rise_min
            and monotonic_decreasing(train_vals)
            and monotonic_increasing(val_vals)
        )

    if train_vals and rec_vals:
        rec_rise = rec_vals[-1] - rec_vals[0]
        overfit_reconst = (
            train_drop is not None
            and train_drop >= train_drop_min
            and rec_rise >= reconst_rise_min
            and monotonic_decreasing(train_vals)
            and monotonic_increasing(rec_vals)
        )

    if train_vals and val_vals and rec_vals:
        plateau = (
            abs(train_vals[-1] - train_vals[0]) <= plateau_train_eps
            and abs(val_vals[-1] - val_vals[0]) <= plateau_val_eps
            and abs(rec_vals[-1] - rec_vals[0]) <= plateau_reconst_eps
        )

    if overfit_val or overfit_reconst:
        verdict = "would_stop_overfit"
        reason = "counterfactual stop: overfit pattern on recent val points"
    elif plateau:
        verdict = "would_review_plateau"
        reason = "counterfactual review: offline metrics plateau"
    else:
        verdict = "continue"
        reason = "no counterfactual stop signal"

    return {
        "status": "active",
        "reason": reason,
        "verdict": verdict,
        "last_epoch": last_epoch,
        "last_global_step": last_step,
        "last_train_loss": last_train,
        "last_val_loss": last.get("val_loss"),
        "last_reconst_mse": last.get("test_reconst_mse"),
        "recent_eval_epochs": [p.epoch for p in recent],
        "recent_train_loss": train_vals,
        "recent_val_loss": val_vals,
        "recent_reconst_mse": rec_vals,
        "delta_train_loss": train_drop,
        "delta_val_loss": val_rise,
        "delta_reconst_mse": rec_rise,
        "signals": {
            "overfit_val": overfit_val,
            "overfit_reconst": overfit_reconst,
            "offline_plateau": plateau,
            "counterfactual_only": True,
            "training_not_stopped": True,
        },
    }


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_markdown_summary(
    md_path: Path,
    run_dir: Path,
    checks: List[Dict[str, Any]],
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Counterfactual Early-Stop Report")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Updated: `{utc_now_iso()}`")
    lines.append("- Mode: `counterfactual_only` (no real stop, no kill signal)")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Time (UTC) | Epoch | Verdict | Reason |")
    lines.append("|---|---:|---|---|")
    for c in checks[-30:]:
        t = c.get("check_time_utc", "")
        e = c.get("last_epoch", "")
        v = c.get("verdict", c.get("status", ""))
        r = str(c.get("reason", "")).replace("|", "/")
        lines.append(f"| {t} | {e} | {v} | {r} |")
    lines.append("")

    # Trigger statistics over full history
    counts: Dict[str, int] = {}
    for c in checks:
        v = str(c.get("verdict", c.get("status", "unknown")))
        counts[v] = counts.get(v, 0) + 1
    lines.append("## Trigger Counts")
    lines.append("")
    for k in sorted(counts):
        lines.append(f"- `{k}`: {counts[k]}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_checks(report_jsonl: Path) -> List[Dict[str, Any]]:
    return read_jsonl(report_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual early-stop watcher. "
            "Writes report only; never stops training."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Hydra run directory")
    parser.add_argument("--interval-sec", type=int, default=3600)
    parser.add_argument("--min-epoch", type=int, default=0)
    parser.add_argument("--val-window", type=int, default=3)
    parser.add_argument("--train-drop-min", type=float, default=0.02)
    parser.add_argument("--val-rise-min", type=float, default=0.05)
    parser.add_argument("--reconst-rise-min", type=float, default=0.005)
    parser.add_argument("--plateau-train-eps", type=float, default=0.02)
    parser.add_argument("--plateau-val-eps", type=float, default=0.05)
    parser.add_argument("--plateau-reconst-eps", type=float, default=0.002)
    parser.add_argument("--once", action="store_true", help="Run one check and exit")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    logs_path = run_dir / "logs.json"
    report_jsonl = run_dir / "early_stop_report.jsonl"
    report_md = run_dir / "early_stop_report.md"

    print(f"[watch] run_dir={run_dir}")
    print(f"[watch] logs={logs_path}")
    print(f"[watch] report_jsonl={report_jsonl}")
    print(f"[watch] report_md={report_md}")
    print(
        "[watch] counterfactual_only=True, training_not_stopped=True, "
        f"min_epoch={args.min_epoch}, interval_sec={args.interval_sec}"
    )

    while True:
        rows = read_jsonl(logs_path)
        result = evaluate_counterfactual(
            rows=rows,
            min_epoch=args.min_epoch,
            val_window=args.val_window,
            train_drop_min=args.train_drop_min,
            val_rise_min=args.val_rise_min,
            reconst_rise_min=args.reconst_rise_min,
            plateau_train_eps=args.plateau_train_eps,
            plateau_val_eps=args.plateau_val_eps,
            plateau_reconst_eps=args.plateau_reconst_eps,
        )
        result["check_time_utc"] = utc_now_iso()
        result["counterfactual_only"] = True
        result["training_not_stopped"] = True
        if isinstance(result.get("last_epoch"), int):
            result["confidence"] = "low" if result["last_epoch"] < 150 else "high"
        else:
            result["confidence"] = "unknown"

        append_jsonl(report_jsonl, result)
        checks = load_existing_checks(report_jsonl)
        write_markdown_summary(report_md, run_dir, checks)

        epoch = result.get("last_epoch")
        verdict = result.get("verdict", result.get("status"))
        reason = result.get("reason")
        print(f"[watch] epoch={epoch} verdict={verdict} reason={reason}")

        if args.once:
            break
        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
