#!/usr/bin/env python3
"""
Aggregate metrics from TrainPolicyWorkspace logs.json across Hydra run directories.

Usage:
  python scripts/summarize_training_runs.py --root output/long
  python scripts/summarize_training_runs.py --root output/long --out output/long/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _hydra_overrides(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / ".hydra" / "overrides.yaml"
    if not p.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text())
        if isinstance(data, list):
            out = {}
            for item in data:
                if "=" in str(item):
                    k, v = str(item).split("=", 1)
                    out[k] = v
            return out
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _find_checkpoints(run_dir: Path) -> List[str]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted([p.name for p in ckpt_dir.glob("*.ckpt")])


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "logs.json")
    overrides = _hydra_overrides(run_dir)

    train = [r["train_loss"] for r in rows if "train_loss" in r]
    val = [r["val_loss"] for r in rows if "val_loss" in r]
    mse = [r["test_reconst_mse"] for r in rows if "test_reconst_mse" in r]
    sr = [r["mean_success_rate"] for r in rows if "mean_success_rate" in r]

    last = rows[-1] if rows else {}
    best_sr = max(sr) if sr else None
    best_sr_epoch = None
    if sr:
        for r in rows:
            if r.get("mean_success_rate") == best_sr:
                best_sr_epoch = r.get("epoch")
                break

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "n_log_rows": len(rows),
        "last_epoch": last.get("epoch"),
        "last_global_step": last.get("global_step"),
        "last_train_loss": train[-1] if train else None,
        "min_train_loss": min(train) if train else None,
        "last_val_loss": val[-1] if val else None,
        "min_val_loss": min(val) if val else None,
        "last_reconst_mse": mse[-1] if mse else None,
        "min_reconst_mse": min(mse) if mse else None,
        "last_mean_success_rate": sr[-1] if sr else None,
        "best_mean_success_rate": best_sr,
        "best_sr_epoch": best_sr_epoch,
        "n_checkpoints": len(_find_checkpoints(run_dir)),
        "embed_dim": overrides.get("policy.embed_dim"),
        "dense_feature_dim": overrides.get("policy.dense_feature_dim"),
        "use_task_uid": overrides.get("policy.use_task_uid_in_state_tokens"),
        "use_dense_visual_memory": overrides.get("policy.use_dense_visual_memory"),
    }


def find_run_dirs(root: Path) -> List[Path]:
    dirs = set()
    for logs in root.rglob("logs.json"):
        dirs.add(logs.parent)
    return sorted(dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OAT policy training runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/long"),
        help="Directory to search for logs.json (recursive).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write summary.csv and summary.md here (default: <root>/summary).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (args.out or (root / "summary")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"No logs.json found under {root}")
        return

    records = [summarize_run(d) for d in run_dirs]
    df = pd.DataFrame(records)
    csv_path = out_dir / "all_runs_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# Training runs summary",
        "",
        f"Root: `{root}`",
        f"Runs found: {len(records)}",
        "",
        "## Table",
        "",
    ]
    md_lines.append("| " + " | ".join(df.columns) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    md_lines.extend(["", "## Per-run logs & checkpoints", ""])
    for rec in records:
        md_lines.append(f"### `{rec['run_name']}`")
        md_lines.append(f"- dir: `{rec['run_dir']}`")
        md_lines.append(f"- logs: `{rec['run_dir']}/logs.json`")
        md_lines.append(f"- checkpoints: `{rec['run_dir']}/checkpoints/`")
        if rec.get("embed_dim"):
            md_lines.append(f"- embed_dim: {rec['embed_dim']}, task_uid: {rec.get('use_task_uid')}")
        md_lines.append("")

    md_path = out_dir / "all_runs_summary.md"
    md_path.write_text("\n".join(md_lines))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    for _, row in df.iterrows():
        print(
            f"  {row['run_name']}: epoch={row['last_epoch']} "
            f"train={row['last_train_loss']} val={row['last_val_loss']} "
            f"sr={row['best_mean_success_rate']}"
        )


if __name__ == "__main__":
    main()
