#!/usr/bin/env python3
"""Publish ep-0950 checkpoint + train/eval artifacts to hackhackhack66666/oat-dense-blt-950."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ID = "hackhackhack66666/oat-dense-blt-950"
TARGET_EPOCH = 950
RUN_NAME = "oat_dense_with_uid_long_0530_220204"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from watch_hf_checkpoint_upload import push_via_git_xet, push_via_hub  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sr_from_eval_log(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    data = load_json(path)
    for key in ("mean_success_rate_mean", "mean_success_rate"):
        if key in data:
            return float(data[key])
    exps = data.get("experiments") or data.get("exp_results") or []
    srs = []
    for e in exps:
        if isinstance(e, dict) and "mean_success_rate" in e:
            srs.append(float(e["mean_success_rate"]))
    return sum(srs) / len(srs) if srs else None


def phase_b_stats(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = load_json(path)
    out: Dict[str, Any] = {}
    for sr_key, err_key in (
        ("mean_success_rate_mean", "mean_success_rate_stderr"),
        ("mean_success_rate", "mean_success_rate_stderr"),
    ):
        if sr_key in data:
            out["mean_sr"] = float(data[sr_key])
            if err_key in data:
                out["mean_sr_stderr"] = float(data[err_key])
            break
    return out


def per_task_table(eval_log: Dict[str, Any]) -> str:
    rows = []
    for k, v in eval_log.items():
        if not k.endswith("/mean_success_rate_mean"):
            continue
        task = k[: -len("/mean_success_rate_mean")]
        short = task[:60] + ("…" if len(task) > 60 else "")
        rows.append((float(v), short))
    rows.sort(reverse=True)
    lines = ["| Task | SR |", "|------|-----|"]
    for sr, name in rows:
        lines.append(f"| `{name}` | {100.0 * sr:.1f}% |")
    return "\n".join(lines)


def readme_md(
    snap: Dict[str, Any],
    sim_eval: Optional[Dict[str, Any]],
    phase_b: Optional[Dict[str, Any]],
    sim_eval_log: Optional[Dict[str, Any]] = None,
    phase_b_log: Optional[Dict[str, Any]] = None,
) -> str:
    tl = snap.get("train_loss", "n/a")
    vl = snap.get("val_loss", "n/a")
    rl = snap.get("test_reconst_mse", "n/a")
    in_loop_sr = snap.get("mean_success_rate")
    sr_cell = (
        f"**{100.0 * float(sim_eval['mean_sr']):.1f}%**"
        if sim_eval and sim_eval.get("mean_sr") is not None
        else "—"
    )
    phase_b_sr = (
        f"**{100.0 * float(phase_b['mean_sr']):.2f}%**"
        if phase_b and phase_b.get("mean_sr") is not None
        else "—"
    )
    phase_b_err = ""
    if phase_b and phase_b.get("mean_sr_stderr") is not None:
        phase_b_err = f" ± {100.0 * float(phase_b['mean_sr_stderr']):.2f}%"

    sim_block = ""
    if sim_eval and sim_eval.get("mean_sr") is not None:
        sr_pct = 100.0 * float(sim_eval["mean_sr"])
        per_task = f"\n\n{per_task_table(sim_eval_log)}\n" if sim_eval_log else ""
        sim_block = f"""
### Sim eval — Phase A screen (30 ep/task)

**Mean success rate: {sr_pct:.1f}%** — 30 episodes/task, 300 total rollouts, seed 1000 (single experiment).

![Phase A sim eval dashboard](sim_eval/sim_eval_dashboard.png)

Details: [`sim_eval/eval_summary.md`](sim_eval/eval_summary.md) · [`sim_eval/eval_log.json`](sim_eval/eval_log.json)
{per_task}"""

    phase_b_block = ""
    if phase_b and phase_b.get("mean_sr") is not None:
        sr_b = 100.0 * float(phase_b["mean_sr"])
        per_task = f"\n\n{per_task_table(phase_b_log)}\n" if phase_b_log else ""
        phase_b_block = f"""
### Sim eval — Phase B confirm (50 ep/task, 3 seeds)

**Mean success rate: {sr_b:.2f}%{phase_b_err}** — official-style protocol for comparison with OAT paper (~56.3%).

Seeds 1000 / 1500 / 2000 (`seed_stride=500`), 500 rollouts total. Per-seed mean SR: **52.0%** · **49.4%** · **50.4%**.

Compared to ep-0700 Phase B confirm (**47.60% ± 1.75%**): **+3.0 pp** on the calibrated protocol.

![Phase B confirm dashboard](sim_eval_phase_b/phase_b_confirm_pt50_ep-0950_dashboard.png)

Details: [`sim_eval_phase_b/eval_summary.md`](sim_eval_phase_b/eval_summary.md) · [`sim_eval_phase_b/eval_log.json`](sim_eval_phase_b/eval_log.json)
{per_task}"""

    in_loop_note = ""
    if in_loop_sr is not None:
        in_loop_note = (
            f"\nIn-loop eval at ep 950 (same run, 30 ep/task): "
            f"**{100.0 * float(in_loop_sr):.2f}%** — noisier than Phase B; "
            f"use Phase B as the calibrated reference.\n"
        )

    return f"""---
license: mit
tags:
- robotics
- libero
- oat
- dense-visual-memory
- imitation-learning
---

# OAT Dense LIBERO-10 — Checkpoint Epoch {TARGET_EPOCH}

Hugging Face model repository for a **dense cross-attention OAT policy** trained on
**LIBERO-10 (N500)**. Snapshot at **epoch {TARGET_EPOCH}** during run `{RUN_NAME}`.

This checkpoint is the best point on the training ladder after resume from ep-0700
(in-loop peak **52.67%**). **Phase B confirm: 50.60% ± 0.76%** on LIBERO-10.

## Quick download

```bash
huggingface-cli download hackhackhack66666/oat-dense-blt-950 ep-0950.ckpt \\
  --local-dir ./checkpoints
```

## Files

| File | Description |
|------|-------------|
| `ep-0950.ckpt` | PyTorch workspace checkpoint (~729 MB) |
| `training_logs.jsonl` | Full training JSONL (train/val curves) |
| `training_metrics_dashboard.png` | Training loss / val / reconst MSE dashboard |
| `training_snapshot_ep0950.json` | Offline metrics at epoch 950 |
| `sim_eval/` | Phase A screen eval (30 ep/task) |
| `sim_eval_phase_b/` | Phase B confirm eval (50 ep/task, 3 seeds) |
| `experiment_log_dense_visual_memory.md` | Experiment journal |

## Metrics @ epoch {TARGET_EPOCH}

| Train loss | Val loss | Reconst MSE | Phase A SR | Phase B SR (confirm) |
|-----------:|---------:|------------:|-----------:|---------------------:|
| {tl} | {vl} | {rl} | {sr_cell} | {phase_b_sr}{phase_b_err} |
{in_loop_note}
## Visualizations

### Training (offline)

![Training metrics dashboard](training_metrics_dashboard.png)
{sim_block}
{phase_b_block}
## Ladder context

| Epoch | Phase A SR | Phase B confirm | HF repo |
|------:|-----------:|----------------:|---------|
| 300 | 39.7% | — | [OAT-BLT-LIBERO-300](https://huggingface.co/hackhackhack66666/OAT-BLT-LIBERO-300) |
| 500 | 38.0% | — | [OAT-BLT-LIBERO-500](https://huggingface.co/hackhackhack66666/OAT-BLT-LIBERO-500) |
| 700 | 51.7% | 47.60% ± 1.75% | [OAT-BLT-Libero-700](https://huggingface.co/hackhackhack66666/OAT-BLT-Libero-700) |
| **950** | **52.7%** | **50.60% ± 0.76%** | **this repo** |

Paper **OAT8** on LIBERO-10: **~56.3%** mean success rate (external reference).

## Model configuration

- **Policy:** OAT with `use_dense_visual_memory=true` (spatial visual tokens + cross-attn)
- **State memory:** enabled (`use_state_memory_tokens=true`)
- **Task UID:** enabled in state tokens
- **Dataset:** [`libero10_N500.zarr`](https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr)
- **Embed dim:** 256
- **Tokenizer:** `tokenizer_ep-0950_mse-0.002.ckpt` (Mirageinv/oat)

## Citation

If you use this checkpoint, please cite **OAT: Ordered Action Tokenization** and specify
epoch **{TARGET_EPOCH}** of the dense LIBERO-10 ladder.

Source: [GadzhiAskhabaliev/OAT-BLT-Dense](https://github.com/GadzhiAskhabaliev/OAT-BLT-Dense) (`BLT-OAT-dense` branch).
"""


def collect_files(repo_root: Path, run_dir: Path, docs_dir: Path) -> Dict[str, Path]:
    ckpt_src = run_dir / "checkpoints" / "ep-0950_sr-0.527.ckpt"
    if not ckpt_src.is_file():
        alt = run_dir / "checkpoints" / "ep-0950.ckpt"
        ckpt_src = alt if alt.is_file() else ckpt_src

    ladder = docs_dir / "ladder_950"
    phase_b_docs = docs_dir / "phase_b_confirm_ep0950"
    phase_b_dash = docs_dir / "phase_b_confirm" / "phase_b_confirm_ep-0950_dashboard.png"
    confirm_eval = (
        repo_root / "output/eval/ladder_confirm_pt50/ep-0950_sr-0.527/eval_log.json"
    )
    if not confirm_eval.is_file():
        confirm_eval = phase_b_docs / "eval_log.json"
    confirm_log = run_dir / "confirm_ep0950_eval.log"

    files: Dict[str, Path] = {
        "ep-0950.ckpt": ckpt_src,
        "training_logs.jsonl": run_dir / "logs.json",
        "training_metrics_dashboard.png": ladder / "training_metrics_dashboard.png",
        "training_snapshot_ep0950.json": ladder / "training_snapshot.json",
        "sim_eval/eval_log.json": ladder / "eval_log.json",
        "sim_eval/eval_summary.md": ladder / "eval_summary.md",
        "sim_eval/sim_eval_dashboard.png": ladder / "sim_eval_dashboard.png",
        "sim_eval_phase_b/eval_log.json": confirm_eval,
        "sim_eval_phase_b/eval_summary.md": phase_b_docs / "eval_summary.md",
        "sim_eval_phase_b/phase_b_confirm_pt50_ep-0950_dashboard.png": phase_b_dash,
        "experiment_log_dense_visual_memory.md": repo_root / "docs" / "experiment_log_dense_visual_memory.md",
    }
    if confirm_log.is_file():
        files["sim_eval_phase_b/confirm_ep0950_eval.log"] = confirm_log

    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/long/oat_dense_with_uid_long_0530_220204"),
    )
    parser.add_argument("--hf-repo", default=REPO_ID)
    parser.add_argument("--hf-repo-dir", type=Path, default=Path("/tmp/hf_oat_dense_blt_950"))
    parser.add_argument("--method", choices=("git-xet", "hub"), default="hub")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload README + visualizations only (skip checkpoint and training_logs)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = (repo_root / args.run_dir).resolve()
    docs_dir = repo_root / "docs" / "results"

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token and not args.dry_run:
        raise SystemExit("HF_TOKEN is not set")

    files = collect_files(repo_root, run_dir, docs_dir)

    snap_path = files["training_snapshot_ep0950.json"]
    sim_log_path = files["sim_eval/eval_log.json"]
    phase_b_log_path = files["sim_eval_phase_b/eval_log.json"]

    missing = [k for k, p in files.items() if not p.is_file()]
    if args.readme_only:
        skip = {"ep-0950.ckpt", "training_logs.jsonl", "sim_eval_phase_b/confirm_ep0950_eval.log"}
        missing = [k for k in missing if k not in skip]
    if missing:
        raise FileNotFoundError(
            "Missing upload sources:\n" + "\n".join(f"  {k}: {files[k]}" for k in missing)
        )

    snap = load_json(snap_path)
    sim_log = load_json(sim_log_path)
    phase_b_log = load_json(phase_b_log_path)
    sim_sr = mean_sr_from_eval_log(sim_log_path)
    phase_b = phase_b_stats(phase_b_log_path)

    readme_path = repo_root / "output/hf_ep0950_staging/README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        readme_md(
            snap,
            {"mean_sr": sim_sr} if sim_sr is not None else None,
            phase_b,
            sim_log,
            phase_b_log,
        ),
        encoding="utf-8",
    )
    files["README.md"] = readme_path

    if args.readme_only:
        keep = {
            "README.md",
            "training_metrics_dashboard.png",
            "training_snapshot_ep0950.json",
            "sim_eval/eval_log.json",
            "sim_eval/eval_summary.md",
            "sim_eval/sim_eval_dashboard.png",
            "sim_eval_phase_b/eval_log.json",
            "sim_eval_phase_b/eval_summary.md",
            "sim_eval_phase_b/phase_b_confirm_pt50_ep-0950_dashboard.png",
            "experiment_log_dense_visual_memory.md",
        }
        files = {k: v for k, v in files.items() if k in keep}

    print(f"[publish-950] {len(files)} files -> {args.hf_repo} ({args.method})")
    for dest, src in sorted(files.items()):
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  {dest} ({size_mb:.1f} MB)")

    if args.dry_run:
        return

    msg = f"OAT dense ep-0950: README + visualizations ({RUN_NAME})"
    if args.method == "git-xet":
        push_via_git_xet(args.hf_repo, args.hf_repo_dir.resolve(), files, msg, hf_token)
    else:
        push_via_hub(args.hf_repo, files, hf_token)
    print("[publish-950] done")


if __name__ == "__main__":
    main()
