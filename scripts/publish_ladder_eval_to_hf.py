#!/usr/bin/env python3
"""Build sim-eval dashboard + upload eval artifacts to the matching HF ladder repo."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

BG = "#0b0f14"
PANEL = "#121820"
GRID = "#2a3544"
CYAN = "#5ec8e8"
PINK = "#ff6b9d"
GOLD = "#f0c040"
WHITE = "#e8eef5"
TEAL = "#3dd6c3"

REPO_BY_TAG = {
    "ep-0300": "hackhackhack66666/OAT-BLT-LIBERO-300",
    "ep-0500": "hackhackhack66666/OAT-BLT-LIBERO-500",
    "ep-0700": "hackhackhack66666/OAT-BLT-Libero-700",
}


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.45, linestyle="--", linewidth=0.6)


def per_task_sr(eval_log: Dict[str, Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for key, val in eval_log.items():
        if key.endswith("/mean_success_rate_mean"):
            task = key[: -len("/mean_success_rate_mean")]
        elif key.endswith("/mean_success_rate") and "/" in key:
            task = key[: -len("/mean_success_rate")]
        else:
            continue
        rows.append((task, float(val)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def make_eval_dashboard(eval_dir: Path, out_png: Path, tag: str) -> Dict[str, Any]:
    log_path = eval_dir / "eval_log.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing {log_path}")
    ev = json.loads(log_path.read_text(encoding="utf-8"))

    sr = ev.get("mean_success_rate_mean", ev.get("mean_success_rate"))
    stderr = ev.get("mean_success_rate_stderr", ev.get("mean_success_rate_std"))
    n_test = ev.get("n_test", "?")
    suite = ev.get("task_suite", "libero10")
    tasks = per_task_sr(ev)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 2.2], hspace=0.35)

    ax_head = fig.add_subplot(gs[0])
    ax_head.set_facecolor(PANEL)
    ax_head.axis("off")
    sr_txt = f"{100.0 * float(sr):.2f}%" if sr is not None else "n/a"
    err_txt = ""
    if stderr is not None and ev.get("num_exp", 1) > 1:
        err_txt = f" ± {100.0 * float(stderr):.2f}%"
    summary = textwrap.dedent(
        f"""
        LIBERO-10 sim eval — {tag}
        mean_success_rate = {sr_txt}{err_txt}
        n_test = {n_test}  |  suite = {suite}
        checkpoint = {ev.get('checkpoint', tag)}
        test_start_seed = {ev.get('test_start_seed', '?')}  |  num_exp = {ev.get('num_exp', 1)}
        """
    ).strip()
    ax_head.text(
        0.02,
        0.85,
        summary,
        transform=ax_head.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color=WHITE,
        family="monospace",
    )

    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(PANEL)
    if tasks:
        labels = [t[:48] + ("…" if len(t) > 48 else "") for t, _ in tasks]
        vals = [100.0 * v for _, v in tasks]
        y = np.arange(len(labels))
        colors = [TEAL if v >= 50 else PINK if v < 30 else CYAN for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.9, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(56.3, color=GOLD, ls="--", lw=1.8, alpha=0.85, label="OAT8 ref ~56.3%")
        if sr is not None:
            ax.axvline(100.0 * float(sr), color=GOLD, ls="-", lw=2.0, alpha=0.55, label="Mean SR")
    ax.set_xlabel("Success rate (%)", fontsize=10, color=WHITE)
    ax.set_title("Per-task success rate", fontsize=12, fontweight="bold", color=WHITE, pad=10)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return {"mean_success_rate": sr, "n_test": n_test, "n_tasks": len(tasks)}


def write_eval_summary(eval_dir: Path, tag: str, phase_label: str = "") -> Path:
    log_path = eval_dir / "eval_log.json"
    ev = json.loads(log_path.read_text(encoding="utf-8"))
    sr = ev.get("mean_success_rate_mean", ev.get("mean_success_rate"))
    stderr = ev.get("mean_success_rate_stderr")
    title = f"Sim eval — {tag}"
    if phase_label:
        title += f" ({phase_label})"
    lines = [
        f"# {title}",
        "",
        f"- **phase**: `{phase_label or 'phase_a_screen'}`",
        f"- **mean_success_rate**: `{sr}`"
        + (f" ± `{stderr}`" if stderr is not None else ""),
        f"- **n_test**: {ev.get('n_test')}",
        f"- **n_test_per_task**: {ev.get('n_test_per_task')}",
        f"- **test_start_seed**: {ev.get('test_start_seed')}",
        f"- **num_exp**: {ev.get('num_exp')}",
        "",
        "## Per-task SR",
        "",
        "| Task | SR |",
        "|------|-----|",
    ]
    for task, val in per_task_sr(ev):
        lines.append(f"| `{task[:60]}` | {100.0 * val:.1f}% |")
    out = eval_dir / "eval_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_hf_readme(eval_dir: Path, tag: str, phase_label: str, hf_prefix: str) -> Path:
    body = textwrap.dedent(
        f"""\
        # {phase_label}

        - **checkpoint**: `{tag}`
        - **HF folder**: `{hf_prefix}/`
        - **Phase A screen** artifacts (if present) live under `sim_eval/` and are not overwritten.

        | File | Description |
        |------|-------------|
        | `{phase_label}_eval_log.json` | Full metrics JSON |
        | `{phase_label}_dashboard.png` | Per-task SR plot |
        | `{phase_label}_summary.md` | Human-readable table |
        | `{phase_label}_run.log` | Eval stdout |
        """
    ).strip()
    out = eval_dir / "hf_upload_readme.md"
    out.write_text(body + "\n", encoding="utf-8")
    return out


def upload_eval_artifacts(
    repo_id: str,
    eval_dir: Path,
    run_log: Path | None,
    hf_token: str,
    hf_prefix: str = "sim_eval",
    artifact_label: str = "phase_a_screen_pt30",
    legacy_hf_layout: bool = False,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    prefix = hf_prefix.strip("/")
    label = artifact_label.strip()
    if legacy_hf_layout:
        files: Dict[str, Path] = {
            f"{prefix}/eval_log.json": eval_dir / "eval_log.json",
            f"{prefix}/sim_eval_dashboard.png": eval_dir / "sim_eval_dashboard.png",
            f"{prefix}/eval_summary.md": eval_dir / "eval_summary.md",
        }
        if run_log is not None and run_log.is_file():
            files[f"{prefix}/eval_run.log"] = run_log
    else:
        files = {
            f"{prefix}/{label}_eval_log.json": eval_dir / "eval_log.json",
            f"{prefix}/{label}_dashboard.png": eval_dir / "sim_eval_dashboard.png",
            f"{prefix}/{label}_summary.md": eval_dir / "eval_summary.md",
            f"{prefix}/README.md": eval_dir / "hf_upload_readme.md",
        }
        if run_log is not None and run_log.is_file():
            files[f"{prefix}/{label}_run.log"] = run_log

    for path_in_repo, local in files.items():
        if not local.is_file():
            raise FileNotFoundError(local)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add sim eval logs and dashboard ({path_in_repo})",
        )
        print(f"  uploaded {path_in_repo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True, choices=list(REPO_BY_TAG))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--run-log", type=Path, default=None)
    parser.add_argument(
        "--hf-prefix",
        default="sim_eval",
        help="HF repo subfolder (Phase A: sim_eval, Phase B: sim_eval_phase_b)",
    )
    parser.add_argument(
        "--artifact-label",
        default="phase_a_screen_pt30",
        help="Distinct filename prefix on HF (e.g. phase_b_confirm_pt50_ep-0700)",
    )
    parser.add_argument("--phase-label", default="", help="Human label for summary/dashboard")
    parser.add_argument(
        "--legacy-hf-layout",
        action="store_true",
        help="Upload Phase A filenames (sim_eval_dashboard.png etc.)",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Upload only dashboard + summary (fix empty plots without touching eval_log)",
    )
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    repo_id = args.repo_id or REPO_BY_TAG[args.tag]
    png = eval_dir / "sim_eval_dashboard.png"
    phase_label = args.phase_label or args.artifact_label

    stats = make_eval_dashboard(eval_dir, png, args.tag)
    write_eval_summary(eval_dir, args.tag, phase_label=phase_label)
    if not args.legacy_hf_layout:
        write_hf_readme(eval_dir, args.tag, phase_label, args.hf_prefix.strip("/"))
    print(f"dashboard: {png}  SR={stats.get('mean_success_rate')} n_tasks={stats.get('n_tasks')}")

    if args.skip_upload:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    print(f"uploading to {repo_id} prefix={args.hf_prefix} ...")
    if args.dashboard_only:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        prefix = args.hf_prefix.strip("/")
        dash_name = "sim_eval_dashboard.png" if args.legacy_hf_layout else f"{args.artifact_label}_dashboard.png"
        sum_name = "eval_summary.md" if args.legacy_hf_layout else f"{args.artifact_label}_summary.md"
        for path_in_repo, local in [
            (f"{prefix}/{dash_name}", png),
            (f"{prefix}/{sum_name}", eval_dir / "eval_summary.md"),
        ]:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Fix sim eval dashboard ({path_in_repo})",
            )
            print(f"  uploaded {path_in_repo}")
        return
    upload_eval_artifacts(
        repo_id,
        eval_dir,
        args.run_log,
        token,
        args.hf_prefix,
        args.artifact_label,
        args.legacy_hf_layout,
    )
    print("done", repo_id)


if __name__ == "__main__":
    main()
