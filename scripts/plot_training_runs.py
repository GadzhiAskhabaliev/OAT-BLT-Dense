#!/usr/bin/env python3
"""
Plot train/val/reconst/SR curves from one or more training run directories.

Usage:
  python scripts/plot_training_runs.py \\
    output/long/oat_dense_with_uid_long_0530_220204 \\
    output/long/dense_emb128_with_uid_053101_000000 \\
    --out output/long/summary/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_logs(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "logs.json"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def series(rows: List[Dict[str, Any]], key: str, x_key: str = "epoch"):
    xs, ys = [], []
    for r in rows:
        if key in r:
            xs.append(r.get(x_key, len(xs)))
            ys.append(r[key])
    return xs, ys


def plot_runs(run_dirs: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("train_loss", "Train loss"),
        ("val_loss", "Validation loss"),
        ("test_reconst_mse", "Reconstruction MSE"),
        ("mean_success_rate", "Mean success rate"),
    ]

    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = False
        for run_dir in run_dirs:
            logs_path = run_dir / "logs.json"
            if not logs_path.is_file():
                continue
            rows = load_logs(run_dir)
            xs, ys = series(rows, metric)
            if not ys:
                continue
            any_data = True
            ax.plot(xs, ys, label=run_dir.name, alpha=0.9)
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / f'{metric}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Hydra run directories containing logs.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/long/summary/plots"),
    )
    args = parser.parse_args()
    run_dirs = [p.expanduser().resolve() for p in args.run_dirs]
    plot_runs(run_dirs, args.out.expanduser().resolve())


if __name__ == "__main__":
    main()
