#!/usr/bin/env python3
"""
Policy-level integration checks for Blockwise-OAT.

Checks:
1) API compatibility: AR and Blockwise outputs have matching shapes.
2) Fallback behavior when blockwise requested but decoder not attached.
3) Inference benchmark speedup on policy conditioning.
4) Optional sim-eval command hints for AR vs Blockwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oat.blockwise_oat import benchmark_blockwise_vs_ar
from oat.policy.base_policy import BasePolicy


def _load_tail_decoder(policy, tail_checkpoint: str | None, prefix_len: int, device: torch.device):
    tail = policy.build_blockwise_tail_decoder(prefix_len=prefix_len).to(device)
    if tail_checkpoint is not None:
        payload = torch.load(tail_checkpoint, map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        tail.load_state_dict(state_dict, strict=True)
    tail.eval()
    return tail


def _shape_info(out: Dict[str, torch.Tensor]) -> Dict[str, list]:
    return {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Blockwise-OAT policy integration")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--tail-checkpoint", default=None, help="Optional trained tail decoder checkpoint (.pt)")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-repeats", type=int, default=8)
    parser.add_argument(
        "--min-param-ratio",
        type=float,
        default=0.35,
        help="Minimum acceptable ParallelTailDecoder/AR parameter ratio",
    )
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    policy, _ = BasePolicy.from_checkpoint(args.policy_checkpoint, return_configuration=True)
    policy.to(device)
    policy.eval()

    obs = policy.create_dummy_observation(batch_size=args.batch_size, device=device)

    # 1) Full AR reference.
    with torch.inference_mode():
        out_ar = policy.predict_action(obs, use_blockwise=False, temperature=0.0, topk=1)

    # 2) Fallback check: force blockwise path without decoder.
    saved_tail = policy.blockwise_tail_decoder
    policy.blockwise_tail_decoder = None
    with torch.inference_mode():
        out_fallback = policy.predict_action(obs, use_blockwise=True, temperature=0.0, topk=1)
    fallback_match = (
        out_fallback["action"].shape == out_ar["action"].shape
        and out_fallback["action_pred"].shape == out_ar["action_pred"].shape
    )

    # 3) Attach decoder and run blockwise.
    policy.blockwise_tail_decoder = _load_tail_decoder(policy, args.tail_checkpoint, args.prefix_len, device)
    tail_params = sum(p.numel() for p in policy.blockwise_tail_decoder.parameters())
    ar_params = sum(p.numel() for p in policy.model.parameters())
    param_ratio = tail_params / max(ar_params, 1)
    with torch.inference_mode():
        out_bw = policy.predict_action(
            obs,
            use_blockwise=True,
            blockwise_prefix_len=args.prefix_len,
            blockwise_refine_iters=args.refine_iters,
            temperature=0.0,
            topk=1,
        )

    # 4) Speed benchmark on same conditioning.
    with torch.inference_mode():
        cond, memory_is_embedded = policy._get_conditioning(obs)
        bench = benchmark_blockwise_vs_ar(
            policy.model,
            policy.blockwise_tail_decoder,
            cond,
            policy.bos_id,
            total_tokens=policy.max_seq_len,
            prefix_len=args.prefix_len,
            memory_is_embedded=memory_is_embedded,
            warmup=args.benchmark_warmup,
            repeats=args.benchmark_repeats,
        )

    policy.blockwise_tail_decoder = saved_tail

    checks = {
        "fallback_shapes_match_ar": bool(fallback_match),
        "blockwise_shapes_match_ar": bool(
            out_bw["action"].shape == out_ar["action"].shape
            and out_bw["action_pred"].shape == out_ar["action_pred"].shape
        ),
        "blockwise_numerics_finite": bool(
            torch.isfinite(out_bw["action"]).all() and torch.isfinite(out_bw["action_pred"]).all()
        ),
        "speedup_gt_1": bool(bench["speedup"] > 1.0),
        "tail_param_ratio_ok": bool(param_ratio >= args.min_param_ratio),
    }

    report = {
        "checkpoint": args.policy_checkpoint,
        "tail_checkpoint": args.tail_checkpoint,
        "prefix_len": args.prefix_len,
        "refine_iters": args.refine_iters,
        "batch_size": args.batch_size,
        "device": str(device),
        "ar_shapes": _shape_info(out_ar),
        "fallback_shapes": _shape_info(out_fallback),
        "blockwise_shapes": _shape_info(out_bw),
        "benchmark": bench,
        "param_counts": {
            "tail_decoder": tail_params,
            "ar_model": ar_params,
            "tail_to_ar_ratio": param_ratio,
            "min_ratio_required": args.min_param_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "sim_eval_commands": {
            "ar": (
                "python scripts/eval_policy_sim.py "
                f"--checkpoint {args.policy_checkpoint} "
                "--output_dir output/eval_blockwise_verify/ar "
                "--n-test-per-task 5 --overwrite"
            ),
            "blockwise": (
                "python scripts/eval_policy_sim.py "
                f"--checkpoint {args.policy_checkpoint} "
                "--output_dir output/eval_blockwise_verify/blockwise "
                "--n-test-per-task 5 --overwrite "
                "--use-blockwise --blockwise-prefix-len "
                f"{args.prefix_len} --blockwise-refine-iters {args.refine_iters}"
            ),
        },
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
