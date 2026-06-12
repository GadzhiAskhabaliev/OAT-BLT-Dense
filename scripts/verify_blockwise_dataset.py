#!/usr/bin/env python3
"""
Dataset validation for Blockwise-OAT.

Checks:
1) Structural/schema sanity on sampled items.
2) Semantic prefix check: tokenize(action) -> detokenize(z1..zP) is valid.
3) Optional smoke subset creation for quick iteration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys
from typing import Dict, List

import hydra
import numpy as np
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oat.policy.base_policy import BasePolicy


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return None


def _ensure_finite(name: str, t: torch.Tensor) -> None:
    if t.is_floating_point() and not torch.isfinite(t).all():
        raise ValueError(f"{name} has non-finite values")


def _create_smoke_subset(src: pathlib.Path, dst: pathlib.Path, episodes: int) -> Dict[str, int]:
    if dst.exists():
        shutil.rmtree(dst)

    src_root = zarr.open(str(src), mode="r")
    ends = np.array(src_root["meta"]["episode_ends"])
    if len(ends) == 0:
        raise ValueError("Source dataset has zero episodes")
    keep_episodes = min(int(episodes), len(ends))
    cutoff = int(ends[keep_episodes - 1])

    dst_root = zarr.open(str(dst), mode="w")
    meta = dst_root.create_group("meta")
    data = dst_root.create_group("data")

    for key, arr in src_root["meta"].items():
        value = np.array(arr)
        if value.ndim >= 1 and value.shape[0] == len(ends):
            value = value[:keep_episodes]
        meta.array(name=key, data=value, chunks=value.shape if value.ndim > 0 else None)

    for key, arr in src_root["data"].items():
        value = np.array(arr[:cutoff])
        chunks = arr.chunks
        if chunks is not None and len(chunks) > 0:
            chunks = (min(chunks[0], value.shape[0]),) + tuple(chunks[1:])
        data.array(name=key, data=value, chunks=chunks)

    return {"episodes": int(keep_episodes), "cutoff": int(cutoff)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset integrity for Blockwise-OAT")
    parser.add_argument("--policy-checkpoint", required=True, help="Path to policy .ckpt")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--schema-samples", type=int, default=256)
    parser.add_argument("--prefix-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-subset-dst", default=None, help="Optional output .zarr path for smoke subset")
    parser.add_argument("--smoke-episodes", type=int, default=32)
    parser.add_argument("--report-json", default=None, help="Optional output path for JSON report")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    policy, cfg = BasePolicy.from_checkpoint(args.policy_checkpoint, return_configuration=True)
    policy.eval()

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    n_schema = min(args.schema_samples, len(dataset))
    sample_indices = rng.sample(range(len(dataset)), n_schema) if n_schema < len(dataset) else list(range(len(dataset)))

    schema_errors: List[str] = []
    expected_obs_keys = set(policy.get_observation_ports())
    action_shapes = set()
    obs_shapes = {}
    obs_dtypes = {}
    action_min, action_max = float("inf"), float("-inf")

    for idx in sample_indices:
        item = dataset[idx]
        if "obs" not in item or "action" not in item:
            schema_errors.append(f"idx={idx}: sample misses 'obs' or 'action'")
            continue

        obs = item["obs"]
        action = _to_tensor(item["action"])
        if action is None:
            schema_errors.append(f"idx={idx}: action is not tensor-like")
            continue

        action_shapes.add(tuple(action.shape))
        _ensure_finite(f"action[{idx}]", action)
        if action.numel() > 0:
            action_min = min(action_min, float(action.min().item()))
            action_max = max(action_max, float(action.max().item()))

        obs_keys = set(obs.keys())
        missing = expected_obs_keys - obs_keys
        if missing:
            schema_errors.append(f"idx={idx}: missing obs keys {sorted(missing)}")

        for key, value in obs.items():
            t = _to_tensor(value)
            if t is None:
                continue
            _ensure_finite(f"obs[{idx}]['{key}']", t)
            obs_shapes.setdefault(key, set()).add(tuple(t.shape))
            obs_dtypes.setdefault(key, set()).add(str(t.dtype))

    # Semantic prefix checks.
    n_prefix = min(args.prefix_samples, len(dataset))
    prefix_indices = rng.sample(range(len(dataset)), n_prefix) if n_prefix < len(dataset) else list(range(len(dataset)))
    prefix_ok = 0
    prefix_errors: List[str] = []

    for idx in prefix_indices:
        action = _to_tensor(dataset[idx]["action"])
        if action is None:
            prefix_errors.append(f"idx={idx}: non-tensor action")
            continue
        act_b = action.unsqueeze(0).to(policy.device)
        with torch.inference_mode():
            tokens = policy.action_tokenizer.tokenize(act_b)
            p = min(args.prefix_len, tokens.shape[1])
            if p <= 0:
                prefix_errors.append(f"idx={idx}: non-positive prefix length after clipping")
                continue
            prefix = tokens[:, :p]
            decoded = policy.action_tokenizer.detokenize(prefix)
        if decoded is None:
            prefix_errors.append(f"idx={idx}: detokenize(prefix) returned None")
            continue
        if not torch.isfinite(decoded).all():
            prefix_errors.append(f"idx={idx}: detokenize(prefix) produced non-finite values")
            continue
        prefix_ok += 1

    zarr_path = pathlib.Path(cfg.task.policy.dataset.zarr_path)
    if not zarr_path.is_absolute():
        zarr_path = ROOT / zarr_path

    smoke_info = None
    if args.smoke_subset_dst:
        smoke_dst = pathlib.Path(args.smoke_subset_dst)
        if not smoke_dst.is_absolute():
            smoke_dst = ROOT / smoke_dst
        smoke_info = _create_smoke_subset(zarr_path, smoke_dst, args.smoke_episodes)
        smoke_info["path"] = str(smoke_dst)

    report = {
        "checkpoint": str(args.policy_checkpoint),
        "dataset_len": int(len(dataset)),
        "dataset_zarr_path": str(zarr_path),
        "schema_samples_checked": int(n_schema),
        "schema_errors": schema_errors,
        "action_shapes_seen": sorted([list(s) for s in action_shapes]),
        "action_min": action_min if action_min != float("inf") else None,
        "action_max": action_max if action_max != float("-inf") else None,
        "obs_shapes_seen": {k: sorted([list(s) for s in v]) for k, v in obs_shapes.items()},
        "obs_dtypes_seen": {k: sorted(list(v)) for k, v in obs_dtypes.items()},
        "prefix_samples_checked": int(n_prefix),
        "prefix_samples_ok": int(prefix_ok),
        "prefix_errors": prefix_errors,
        "smoke_subset": smoke_info,
        "passed": len(schema_errors) == 0 and len(prefix_errors) == 0,
    }

    out = json.dumps(report, indent=2, sort_keys=True)
    print(out)
    if args.report_json:
        report_path = pathlib.Path(args.report_json)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(out + "\n", encoding="utf-8")

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
