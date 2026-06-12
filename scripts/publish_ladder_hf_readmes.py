#!/usr/bin/env python3
"""
Build dark-theme training metric dashboards and publish README + plots to HF ladder repos.

Usage:
  export HF_TOKEN=hf_...
  python scripts/publish_ladder_hf_readmes.py \\
    --run-dir output/long/oat_dense_with_uid_long_0530_220204 \\
    --out output/hf_ladder_publish
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Dashboard palette (dark theme, similar to OAT eval dashboards)
BG = "#0b0f14"
PANEL = "#121820"
GRID = "#2a3544"
CYAN = "#5ec8e8"
PINK = "#ff6b9d"
TEAL = "#3dd6c3"
GOLD = "#f0c040"
WHITE = "#e8eef5"

LADDER: List[Tuple[int, str]] = [
    (300, "hackhackhack66666/OAT-BLT-LIBERO-300"),
    (500, "hackhackhack66666/OAT-BLT-LIBERO-500"),
    (700, "hackhackhack66666/OAT-BLT-Libero-700"),
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def val_series(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_reconst_mse": [],
    }
    for r in rows:
        if "val_loss" not in r:
            continue
        ep = int(r["epoch"])
        out["epoch"].append(float(ep))
        out["train_loss"].append(float(r.get("train_loss", np.nan)))
        out["val_loss"].append(float(r["val_loss"]))
        out["test_reconst_mse"].append(float(r.get("test_reconst_mse", np.nan)))
    return out


def train_epoch_curve(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    by_ep: Dict[int, float] = {}
    for r in rows:
        if "train_loss" not in r or "epoch" not in r:
            continue
        ep = int(r["epoch"])
        by_ep[ep] = float(r["train_loss"])
    xs = sorted(by_ep)
    return [float(x) for x in xs], [by_ep[x] for x in xs]


def sr_series(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        if "mean_success_rate" in r and "epoch" in r:
            xs.append(float(int(r["epoch"])))
            ys.append(100.0 * float(r["mean_success_rate"]))
    return xs, ys


def snapshot_metrics(
    rows: List[Dict[str, Any]], target_epoch: int
) -> Dict[str, Any]:
    for r in reversed(rows):
        if int(r.get("epoch", -1)) != target_epoch:
            continue
        snap: Dict[str, Any] = {"epoch": target_epoch}
        for key in ("train_loss", "val_loss", "test_reconst_mse", "mean_success_rate"):
            if key in r:
                snap[key] = r[key]
        if len(snap) > 1:
            return snap
    return {"epoch": target_epoch}


def load_upload_meta(run_dir: Path, epoch: int) -> Dict[str, Any]:
    p = run_dir / "checkpoints" / f"ep-{epoch:04d}_upload_meta.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.45, linestyle="--", linewidth=0.6)


def make_dashboard(
    run_dir: Path,
    target_epoch: int,
    out_png: Path,
    run_name: str,
) -> Dict[str, Any]:
    rows = load_jsonl(run_dir / "logs.json")
    meta = load_upload_meta(run_dir, target_epoch)
    if meta:
        snap = meta
    else:
        snap = snapshot_metrics(rows, target_epoch)

    val = val_series(rows)
    tr_x, tr_y = train_epoch_curve(rows)
    sr_x, sr_y = sr_series(rows)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], hspace=0.32, wspace=0.28)

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor(PANEL)
    if tr_x:
        ax_main.plot(tr_x, tr_y, color=CYAN, lw=1.8, alpha=0.85, label="Train loss (epoch avg)")
    if val["epoch"]:
        ax_main.plot(
            val["epoch"],
            val["train_loss"],
            color=CYAN,
            ls="",
            marker="o",
            ms=3,
            alpha=0.35,
            label="_nolegend_",
        )
        ax_main.plot(
            val["epoch"],
            val["val_loss"],
            color=PINK,
            lw=2.0,
            marker="o",
            ms=4,
            label="Val loss",
        )
        ax_main2 = ax_main.twinx()
        ax_main2.plot(
            val["epoch"],
            val["test_reconst_mse"],
            color=TEAL,
            lw=1.6,
            marker="s",
            ms=3,
            alpha=0.9,
            label="Reconst MSE",
        )
        ax_main2.set_ylabel("test_reconst_mse", color=TEAL, fontsize=10)
        ax_main2.tick_params(axis="y", colors=TEAL)
        ax_main2.spines["right"].set_color(GRID)
        if sr_x:
            ax_sr = ax_main.twinx()
            if ax_main2 is not ax_sr:
                ax_sr.spines["right"].set_position(("axes", 1.08))
            ax_sr.plot(
                sr_x,
                sr_y,
                color=GOLD,
                lw=0,
                marker="D",
                ms=5,
                alpha=0.95,
                label="In-loop SR (%)",
            )
            ax_sr.set_ylabel("Success rate (%)", color=GOLD, fontsize=10)
            ax_sr.tick_params(axis="y", colors=GOLD)
            ax_sr.spines["right"].set_color(GOLD)

    ax_main.axvline(target_epoch, color=GOLD, ls="--", lw=2.0, alpha=0.95, label=f"Checkpoint ep {target_epoch}")
    ax_main.set_xlabel("Epoch", fontsize=11)
    ax_main.set_ylabel("Loss", fontsize=11)
    ax_main.set_title(
        f"OAT Dense LIBERO-10 — Offline Metrics @ Epoch {target_epoch}\n{run_name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    style_axes(ax_main)
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.25)

    tl = snap.get("train_loss")
    vl = snap.get("val_loss")
    rl = snap.get("test_reconst_mse") or snap.get("last_test_reconst_mse")
    stats = (
        f"Snapshot @ ep {target_epoch}\n"
        f"train_loss = {tl:.4f}\n" if tl is not None else ""
    ) + (
        f"val_loss = {vl:.4f}\n" if vl is not None else ""
    ) + (
        f"reconst_mse = {rl:.5f}\n" if rl is not None else ""
    ) + (
        f"in-loop SR = {100.0 * float(snap['mean_success_rate']):.2f}%\n"
        if snap.get("mean_success_rate") is not None
        else "SR: run sim eval (not in train logs;\nlazy_eval was enabled)"
    )
    ax_main.text(
        0.015,
        0.97,
        stats,
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=WHITE,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#1a2330", edgecolor=GRID, alpha=0.92),
    )

    # Bar chart — snapshot metrics (normalized display for shape)
    ax_bar = fig.add_subplot(gs[1, 0])
    names = []
    values = []
    colors = []
    if tl is not None:
        names.append("train_loss")
        values.append(float(tl))
        colors.append(CYAN)
    if vl is not None:
        names.append("val_loss")
        values.append(float(vl))
        colors.append(PINK)
    if rl is not None:
        names.append("reconst_mse")
        values.append(float(rl))
        colors.append(TEAL)
    if names:
        xpos = np.arange(len(names))
        ax_bar.bar(xpos, values, color=colors, alpha=0.9, edgecolor=WHITE, linewidth=0.4)
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_bar.set_title("Metrics at checkpoint epoch", fontsize=11, pad=8)
    style_axes(ax_bar)

    # Val loss history (distribution of recent val points)
    ax_hist = fig.add_subplot(gs[1, 1])
    if val["val_loss"]:
        recent = val["val_loss"][-30:] if len(val["val_loss"]) > 30 else val["val_loss"]
        ax_hist.hist(recent, bins=min(12, max(4, len(recent) // 2)), color=TEAL, alpha=0.75, edgecolor=WHITE)
        mean_v = float(np.mean(recent))
        med_v = float(np.median(recent))
        ax_hist.axvline(mean_v, color=PINK, ls="--", lw=1.8, label=f"mean={mean_v:.3f}")
        ax_hist.axvline(med_v, color=CYAN, ls=":", lw=1.8, label=f"median={med_v:.3f}")
        if vl is not None:
            ax_hist.axvline(float(vl), color=GOLD, lw=2.0, label=f"snapshot={float(vl):.3f}")
        ax_hist.legend(fontsize=8, framealpha=0.2)
    ax_hist.set_xlabel("Val loss", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Recent val-loss distribution", fontsize=11, pad=8)
    style_axes(ax_hist)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return snap


def load_sim_eval(eval_root: Path, tag: str) -> Optional[Dict[str, Any]]:
    p = eval_root / tag / "eval_log.json"
    if not p.is_file():
        return None
    ev = json.loads(p.read_text(encoding="utf-8"))
    sr = ev.get("mean_success_rate_mean", ev.get("mean_success_rate"))
    return {
        "tag": tag,
        "mean_sr": sr,
        "mean_sr_stderr": ev.get("mean_success_rate_stderr"),
        "n_test": ev.get("n_test"),
        "n_test_per_task": ev.get("n_test_per_task"),
        "num_exp": ev.get("num_exp", 1),
        "test_start_seed": ev.get("test_start_seed"),
    }


def load_phase_b_sim_eval(confirm_root: Path, tag: str) -> Optional[Dict[str, Any]]:
    p = confirm_root / tag / "eval_log.json"
    if not p.is_file():
        return None
    ev = json.loads(p.read_text(encoding="utf-8"))
    sr = ev.get("mean_success_rate_mean", ev.get("mean_success_rate"))
    return {
        "tag": tag,
        "mean_sr": sr,
        "mean_sr_stderr": ev.get("mean_success_rate_stderr"),
        "n_test": ev.get("n_test"),
        "n_test_per_task": ev.get("n_test_per_task"),
        "num_exp": ev.get("num_exp", 3),
        "test_start_seed": ev.get("test_start_seed"),
    }


def readme_md(
    repo_id: str,
    epoch: int,
    snap: Dict[str, Any],
    run_name: str,
    sim_eval: Optional[Dict[str, Any]] = None,
    phase_b: Optional[Dict[str, Any]] = None,
) -> str:
    tl = snap.get("train_loss", "n/a")
    vl = snap.get("val_loss", "n/a")
    rl = snap.get("test_reconst_mse") or snap.get("last_test_reconst_mse", "n/a")
    ckpt = f"ep-{epoch:04d}.ckpt"
    sr_cell = (
        f"**{100.0 * float(sim_eval['mean_sr']):.1f}%**"
        if sim_eval and sim_eval.get("mean_sr") is not None
        else "—"
    )

    sim_block = ""
    if sim_eval and sim_eval.get("mean_sr") is not None:
        sr_pct = 100.0 * float(sim_eval["mean_sr"])
        sim_block = f"""
### Sim eval (LIBERO-10, Phase A screen)

**Mean success rate: {sr_pct:.1f}%** — {sim_eval.get("n_test_per_task", 30)} episodes/task, `{sim_eval.get("n_test", 300)}` total rollouts, seed {sim_eval.get("test_start_seed", 1000)}.
Details: [`sim_eval/eval_summary.md`](sim_eval/eval_summary.md) · [`sim_eval/eval_log.json`](sim_eval/eval_log.json)

![Sim eval dashboard](sim_eval/sim_eval_dashboard.png)
"""

    phase_b_block = ""
    if phase_b and phase_b.get("mean_sr") is not None:
        sr_b = 100.0 * float(phase_b["mean_sr"])
        err = phase_b.get("mean_sr_stderr")
        err_txt = f" ± {100.0 * float(err):.2f}%" if err is not None else ""
        phase_b_block = f"""
### Sim eval Phase B confirm (50 ep/task, 3 seeds)

**Mean success rate: {sr_b:.2f}%{err_txt}** — official-style protocol for comparison with OAT paper (~56.3%).

![Phase B confirm dashboard](sim_eval_phase_b/phase_b_confirm_pt50_ep-{epoch:04d}_dashboard.png)

Details: [`sim_eval_phase_b/`](sim_eval_phase_b/) (separate from Phase A `sim_eval/` — not overwritten).
"""

    # YAML frontmatter must start at column 0 (no dedent/indent).
    frontmatter = """---
license: mit
tags:
- robotics
- libero
- oat
- dense-visual-memory
---
"""

    body = f"""# OAT Dense LIBERO-10 — Checkpoint Epoch {epoch}

Hugging Face model repository for a **dense cross-attention OAT policy** trained on
**LIBERO-10 (N500)**. This snapshot was taken at **epoch {epoch}** during a long run
(`{run_name}`).

## Files

| File | Description |
|------|-------------|
| `{ckpt}` | PyTorch workspace checkpoint (~729 MB) |
| `training_logs.jsonl` | Full training JSONL (train/val curves) |
| `training_metrics_dashboard.png` | Training loss dashboard |
| `overfit_watcher/` | Counterfactual early-stop reports |
| `sim_eval/` | Phase A screen eval (30 ep/task) |
| `sim_eval_phase_b/` | Phase B confirm eval (50 ep/task, 3 exp) — ep-0700 only |
| `experiment_log_dense_visual_memory.md` | Experiment journal |

## Metrics @ epoch {epoch}

| Train loss | Val loss | Reconst MSE | Sim SR (Phase A) |
|-----------:|---------:|------------:|-----------------:|
| {tl} | {vl} | {rl} | {sr_cell} |

### Training (offline)

![Training metrics dashboard](training_metrics_dashboard.png)
{sim_block}
{phase_b_block}
## Model configuration (summary)

- **Policy:** OAT with `use_dense_visual_memory=true` (spatial visual tokens + cross-attn)
- **State memory:** enabled (`use_state_memory_tokens=true`)
- **Task UID:** enabled in state tokens
- **Dataset:** `libero10_N500.zarr`
- **Embed dim:** 256

## Baseline reference

Paper **OAT8** on LIBERO-10: **~56.3%** mean success rate (external reference).

## Citation

If you use this checkpoint, please cite **OAT: Ordered Action Tokenization** and specify
epoch **{epoch}** of the dense LIBERO-10 ladder.
"""
    return frontmatter + body


def publish_repo(
    hf_token: str,
    repo_id: str,
    epoch: int,
    files: Dict[str, Path],
    readme_path: Path,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    upload_map = dict(files)
    upload_map["README.md"] = readme_path
    for dest, src in upload_map.items():
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add training dashboard and README for epoch {epoch}",
        )
    print(f"Published https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/hf_ladder_publish"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload README.md only (use with --eval-root for sim eval section)",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Path to ladder_screen_pt30 for sim eval SR in README",
    )
    parser.add_argument(
        "--confirm-root",
        type=Path,
        default=None,
        help="Path to ladder_confirm_pt50 for Phase B block (ep-0700 repo)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    run_name = run_dir.name
    eval_root = args.eval_root.expanduser().resolve() if args.eval_root else None
    confirm_root = args.confirm_root.expanduser().resolve() if args.confirm_root else None

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token and not args.dry_run:
        raise SystemExit("Set HF_TOKEN=hf_... or pass --dry-run")

    for epoch, repo_id in LADDER:
        repo_out = out_root / repo_id.split("/")[-1]
        png = repo_out / "training_metrics_dashboard.png"
        meta_src = run_dir / "checkpoints" / f"ep-{epoch:04d}_upload_meta.json"
        tag = f"ep-{epoch:04d}"
        sim_eval = load_sim_eval(eval_root, tag) if eval_root else None
        phase_b = (
            load_phase_b_sim_eval(confirm_root, tag)
            if confirm_root and epoch == 700
            else None
        )

        if args.readme_only:
            snap = load_upload_meta(run_dir, epoch) or snapshot_metrics(
                load_jsonl(run_dir / "training_logs.jsonl")
                if (run_dir / "training_logs.jsonl").is_file()
                else load_jsonl(run_dir / "logs.json"),
                epoch,
            )
            readme_path = repo_out / "README.md"
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(
                readme_md(repo_id, epoch, snap, run_name, sim_eval, phase_b),
                encoding="utf-8",
            )
            if args.dry_run:
                print(f"[dry-run] README {repo_id}")
                continue
            from huggingface_hub import HfApi
            HfApi(token=hf_token).upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update README with sim eval (epoch {epoch})",
            )
            print(f"README -> https://huggingface.co/{repo_id}")
            continue

        if args.upload_only:
            if not png.is_file():
                raise SystemExit(f"Missing {png} (run without --upload-only first)")
            snap = load_upload_meta(run_dir, epoch) or {"epoch": epoch}
            readme_path = repo_out / "README.md"
            readme_path.write_text(
                readme_md(repo_id, epoch, snap, run_name, sim_eval, phase_b),
                encoding="utf-8",
            )
        else:
            snap = make_dashboard(run_dir, epoch, png, run_name)
            readme_path = repo_out / "README.md"
            readme_path.write_text(
                readme_md(repo_id, epoch, snap, run_name, sim_eval, phase_b),
                encoding="utf-8",
            )

        if args.dry_run:
            print(f"[dry-run] {repo_id} -> {repo_out}")
            continue

        upload_files: Dict[str, Path] = {"training_metrics_dashboard.png": png}
        if meta_src.is_file():
            upload_files[meta_src.name] = meta_src

        publish_repo(hf_token, repo_id, epoch, upload_files, readme_path)


if __name__ == "__main__":
    main()
