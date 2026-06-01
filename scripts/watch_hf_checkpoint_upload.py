#!/usr/bin/env python3
"""
Watch training logs and upload a checkpoint snapshot to Hugging Face at a target epoch.

This script NEVER stops training. It:
1. Waits until logs.json reports epoch >= target_epoch
2. Copies checkpoints/latest.ckpt -> checkpoints/ep-XXXX.ckpt (snapshot)
3. Pushes the snapshot to an HF model repo (git-xet git push, or huggingface_hub fallback)

Requires HF_TOKEN in the environment for unattended upload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            continue
    return rows


def max_epoch_from_logs(logs_path: Path) -> int:
    rows = read_jsonl(logs_path)
    if not rows:
        return -1
    return max(int(r.get("epoch", -1)) for r in rows)


def epoch_summary(logs_path: Path, target_epoch: int) -> Dict[str, Any]:
    rows = read_jsonl(logs_path)
    summary: Dict[str, Any] = {"target_epoch": target_epoch}
    by_ep: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if "epoch" in r:
            by_ep[int(r["epoch"])] = r
    if target_epoch in by_ep:
        summary.update(by_ep[target_epoch])
    val_rows = [r for r in rows if "val_loss" in r and int(r.get("epoch", -1)) <= target_epoch]
    if val_rows:
        last_val = val_rows[-1]
        summary["last_val_epoch"] = last_val.get("epoch")
        summary["last_val_loss"] = last_val.get("val_loss")
        summary["last_test_reconst_mse"] = last_val.get("test_reconst_mse")
    return summary


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def copy_snapshot(latest_ckpt: Path, snapshot_ckpt: Path) -> None:
    if not latest_ckpt.is_file():
        raise FileNotFoundError(f"latest checkpoint not found: {latest_ckpt}")
    tmp = snapshot_ckpt.with_suffix(".ckpt.part")
    shutil.copy2(latest_ckpt, tmp)
    tmp.replace(snapshot_ckpt)


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True,
        )


def wait_for_fresh_checkpoint(
    latest_ckpt: Path,
    max_age_sec: int = 600,
    timeout_sec: int = 900,
    poll_sec: int = 15,
) -> None:
    """Wait until latest.ckpt exists and was modified recently."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if latest_ckpt.is_file():
            age = time.time() - latest_ckpt.stat().st_mtime
            if age <= max_age_sec:
                print(
                    f"[hf-upload] latest.ckpt age={age:.0f}s "
                    f"(<= {max_age_sec}s) — ready"
                )
                return
        time.sleep(poll_sec)
    raise TimeoutError(
        f"latest.ckpt not fresh within {timeout_sec}s: {latest_ckpt}"
    )


def ensure_git_xet() -> bool:
    try:
        subprocess.run(
            ["git", "xet", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "git-xet"],
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(["git", "xet", "install"], check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def push_via_git_xet(
    hf_repo: str,
    repo_dir: Path,
    files: Dict[str, Path],
    commit_message: str,
    hf_token: str,
) -> None:
    if not ensure_git_xet():
        raise RuntimeError("git-xet is not available and could not be installed")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://user:{hf_token}@huggingface.co/{hf_repo}"

    if not (repo_dir / ".git").is_dir():
        subprocess.run(
            ["git", "clone", clone_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--rebase", "--autostash"],
            check=False,
            capture_output=True,
            text=True,
        )

    for dest_name, src_path in files.items():
        if not src_path.is_file():
            raise FileNotFoundError(f"upload source missing: {src_path}")
        shutil.copy2(src_path, repo_dir / dest_name)

    subprocess.run(["git", "-C", str(repo_dir), "add"] + list(files.keys()), check=True)
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        print("[hf-upload] git: nothing to commit (already up to date)")
        return

    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", commit_message],
        check=True,
        capture_output=True,
        text=True,
    )
    push = subprocess.run(
        ["git", "-C", str(repo_dir), "push"],
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        raise RuntimeError(
            f"git push failed ({push.returncode}): {push.stderr or push.stdout}"
        )


def push_via_hub(
    hf_repo: str,
    files: Dict[str, Path],
    hf_token: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True,
        )
        from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
    for dest_name, src_path in files.items():
        api.upload_file(
            path_or_fileobj=str(src_path),
            path_in_repo=dest_name,
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"upload {dest_name}",
        )


def upload_checkpoint(
    run_dir: Path,
    target_epoch: int,
    hf_repo: str,
    hf_repo_dir: Path,
    method: str,
    post_checkpoint_delay_sec: int,
) -> Dict[str, Any]:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export HF_TOKEN=hf_... before starting the watcher."
        )

    ckpt_dir = run_dir / "checkpoints"
    latest_ckpt = ckpt_dir / "latest.ckpt"
    snapshot_name = f"ep-{target_epoch:04d}.ckpt"
    snapshot_ckpt = ckpt_dir / snapshot_name
    marker = ckpt_dir / f".hf_uploaded_ep{target_epoch}"

    if marker.is_file():
        return {
            "status": "already_uploaded",
            "target_epoch": target_epoch,
            "marker": str(marker),
        }

    # epoch > target_epoch in logs => end-of-epoch ckpt for target_epoch should exist.
    print(
        f"[hf-upload] waiting for latest.ckpt "
        f"(delay={post_checkpoint_delay_sec}s, then freshness poll)..."
    )
    time.sleep(max(0, post_checkpoint_delay_sec))
    wait_for_fresh_checkpoint(latest_ckpt)

    copy_snapshot(latest_ckpt, snapshot_ckpt)
    print(f"[hf-upload] snapshot copied -> {snapshot_ckpt}")

    summary = epoch_summary(run_dir / "logs.json", target_epoch)
    summary["upload_time_utc"] = utc_now_iso()
    summary["hf_repo"] = hf_repo
    summary["snapshot_file"] = snapshot_name
    summary_path = ckpt_dir / f"ep-{target_epoch:04d}_upload_meta.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    files = {
        snapshot_name: snapshot_ckpt,
        summary_path.name: summary_path,
    }

    commit_message = (
        f"OAT dense checkpoint epoch {target_epoch} "
        f"(train continues; snapshot only)"
    )

    if method == "git":
        push_via_git_xet(hf_repo, hf_repo_dir, files, commit_message, hf_token)
        upload_method = "git-xet"
    elif method == "hub":
        push_via_hub(hf_repo, files, hf_token)
        upload_method = "huggingface_hub"
    else:
        try:
            push_via_git_xet(hf_repo, hf_repo_dir, files, commit_message, hf_token)
            upload_method = "git-xet"
        except Exception as git_exc:
            print(f"[hf-upload] git-xet failed ({git_exc}); falling back to huggingface_hub")
            push_via_hub(hf_repo, files, hf_token)
            upload_method = "huggingface_hub"

    marker.write_text(
        json.dumps(
            {
                "target_epoch": target_epoch,
                "upload_time_utc": utc_now_iso(),
                "method": upload_method,
                "hf_repo": hf_repo,
                "snapshot": snapshot_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "status": "uploaded",
        "target_epoch": target_epoch,
        "method": upload_method,
        "hf_repo": hf_repo,
        "snapshot": str(snapshot_ckpt),
        "summary": summary,
    }


def parse_epoch_repo_map(
    epoch_repo_args: List[str],
    target_epochs: List[int],
    default_repo: str,
) -> Dict[int, str]:
    mapping = {te: default_repo for te in target_epochs}
    for item in epoch_repo_args:
        if "=" not in item:
            raise ValueError(f"--epoch-repo must be EPOCH=REPO, got: {item!r}")
        ep_str, repo = item.split("=", 1)
        mapping[int(ep_str.strip())] = repo.strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload checkpoint to Hugging Face at target epoch without stopping training."
    )
    parser.add_argument("--run-dir", required=True, help="Hydra run directory")
    parser.add_argument(
        "--target-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Upload snapshots when logs pass each epoch (e.g. 300 500)",
    )
    parser.add_argument(
        "--target-epoch",
        type=int,
        default=None,
        help="Deprecated: use --target-epochs",
    )
    parser.add_argument(
        "--hf-repo",
        default="hackhackhack66666/OAT-BLT-LIBERO-300",
        help="Default HF repo if --epoch-repo not set for an epoch",
    )
    parser.add_argument(
        "--epoch-repo",
        action="append",
        default=[],
        metavar="EPOCH=REPO",
        help=(
            "Per-epoch HF repo, e.g. "
            "300=hackhackhack66666/OAT-BLT-LIBERO-300 "
            "500=hackhackhack66666/OAT-BLT-LIBERO-500"
        ),
    )
    parser.add_argument(
        "--hf-repo-dir",
        default=None,
        help="Base dir for git-xet clones (default: ~/hf_push/<repo-name>)",
    )
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument(
        "--post-checkpoint-delay-sec",
        type=int,
        default=120,
        help="Wait after epoch threshold so latest.ckpt is flushed",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "git", "hub"],
        default="auto",
        help="Upload method: git-xet, huggingface_hub, or auto",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one check cycle (or when all targets uploaded)",
    )
    parser.add_argument(
        "--exit-when-all-done",
        action="store_true",
        default=True,
        help="Exit loop after every target epoch has been uploaded",
    )
    parser.add_argument(
        "--no-exit-when-all-done",
        action="store_false",
        dest="exit_when_all_done",
        help="Keep polling after all uploads complete",
    )
    args = parser.parse_args()

    if args.target_epochs:
        target_epochs = sorted(set(args.target_epochs))
    elif args.target_epoch is not None:
        target_epochs = [args.target_epoch]
    else:
        target_epochs = [300]

    run_dir = Path(args.run_dir).expanduser().resolve()
    logs_path = run_dir / "logs.json"
    report_jsonl = run_dir / "hf_upload_report.jsonl"
    epoch_repo_map = parse_epoch_repo_map(
        args.epoch_repo, target_epochs, args.hf_repo
    )
    hf_push_base = Path(
        args.hf_repo_dir or Path.home() / "hf_push"
    ).expanduser().resolve()

    def marker_path(te: int) -> Path:
        return run_dir / "checkpoints" / f".hf_uploaded_ep{te}"

    def hf_repo_dir_for(te: int) -> Path:
        repo = epoch_repo_map[te]
        return hf_push_base / repo.split("/")[-1]

    def all_uploaded() -> bool:
        return all(marker_path(te).is_file() for te in target_epochs)

    print(f"[hf-upload] run_dir={run_dir}")
    print(f"[hf-upload] logs={logs_path}")
    print(f"[hf-upload] epoch_repo_map={epoch_repo_map}")
    print(f"[hf-upload] hf_push_base={hf_push_base}")
    print(f"[hf-upload] target_epochs={target_epochs}, method={args.method}")
    print("[hf-upload] training_not_stopped=True")

    if not os.environ.get("HF_TOKEN", "").strip():
        print("[hf-upload] ERROR: HF_TOKEN is not set", file=sys.stderr)
        sys.exit(1)

    ensure_huggingface_hub()

    while True:
        epoch = max_epoch_from_logs(logs_path)
        did_work = False

        for target_epoch in target_epochs:
            marker = marker_path(target_epoch)
            payload: Dict[str, Any] = {
                "check_time_utc": utc_now_iso(),
                "last_epoch": epoch,
                "target_epoch": target_epoch,
                "target_epochs": target_epochs,
                "training_not_stopped": True,
            }

            if marker.is_file():
                continue

            if epoch <= target_epoch:
                payload["status"] = "waiting_target_epoch"
                payload["reason"] = (
                    f"epoch={epoch} <= target_epoch={target_epoch} "
                    f"(need epoch>{target_epoch} for ep-{target_epoch:04d}.ckpt)"
                )
                append_jsonl(report_jsonl, payload)
                print(
                    f"[hf-upload] epoch={epoch} target={target_epoch} "
                    f"status=waiting_target_epoch"
                )
                continue

            snapshot_ckpt = (
                run_dir / "checkpoints" / f"ep-{target_epoch:04d}.ckpt"
            )
            # Avoid uploading current latest.ckpt under a stale target label.
            late_margin = 25
            if (
                epoch > target_epoch + late_margin
                and not snapshot_ckpt.is_file()
            ):
                payload["status"] = "missed_upload_window"
                payload["reason"] = (
                    f"epoch={epoch} >> target={target_epoch} and no local "
                    f"{snapshot_ckpt.name}; refusing to mis-label latest.ckpt"
                )
                append_jsonl(report_jsonl, payload)
                print(
                    f"[hf-upload] epoch={epoch} target={target_epoch} "
                    f"status=missed_upload_window"
                )
                continue

            try:
                hf_repo = epoch_repo_map[target_epoch]
                result = upload_checkpoint(
                    run_dir=run_dir,
                    target_epoch=target_epoch,
                    hf_repo=hf_repo,
                    hf_repo_dir=hf_repo_dir_for(target_epoch),
                    method=args.method,
                    post_checkpoint_delay_sec=args.post_checkpoint_delay_sec,
                )
                payload.update(result)
                append_jsonl(report_jsonl, payload)
                did_work = True
                print(
                    f"[hf-upload] epoch={epoch} target={target_epoch} "
                    f"status={result['status']} method={result.get('method')}"
                )
            except Exception as exc:
                payload["status"] = "upload_failed"
                payload["error"] = str(exc)
                append_jsonl(report_jsonl, payload)
                print(
                    f"[hf-upload] epoch={epoch} target={target_epoch} "
                    f"status=upload_failed error={exc}",
                    file=sys.stderr,
                )
                if args.once:
                    sys.exit(1)

        if all_uploaded():
            print(f"[hf-upload] all targets uploaded: {target_epochs}")
            if args.exit_when_all_done:
                break

        if args.once:
            break

        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
