#!/usr/bin/env python3
"""
Training-loop verification for Blockwise-OAT.

Checks:
1) Micro overfit on synthetic mapping (tail decoder only).
2) Reference loss consistency: training_loss vs manual CE.
3) Refinement stability diagnostics for refine_iters > 1.
4) Optional one-batch real-data sanity from a policy checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict

import hydra
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oat.blockwise_oat import ParallelTailDecoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.policy.base_policy import BasePolicy


def _tiny_ar_model(vocab_size: int, d_model: int, max_seq: int) -> AutoregressiveModel:
    return AutoregressiveModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        max_cond_len=4,
        max_memory_len=4,
        cond_dim=d_model,
        n_layer=2,
        n_head=4,
        n_emb=d_model,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )


def run_micro_overfit(
    vocab_size: int,
    d_model: int,
    prefix_len: int,
    n_tail: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: torch.device,
    refine_iters: int,
) -> Dict:
    ar = _tiny_ar_model(vocab_size=vocab_size, d_model=d_model, max_seq=prefix_len + n_tail + 1).to(device)
    tail = ParallelTailDecoder(vocab_size=vocab_size, d_model=d_model, n_tail=n_tail, n_layers=2).to(device)
    tail.train()
    opt = torch.optim.AdamW(tail.parameters(), lr=lr)

    # Synthetic mapping: target tail is deterministic function of prefix.
    prefix_hidden = torch.randn(batch_size, d_model, device=device)
    prefix_tokens = torch.randint(0, vocab_size - 1, (batch_size, prefix_len), device=device)
    base = prefix_tokens.sum(dim=1, keepdim=True) % (vocab_size - 1)
    offsets = torch.arange(1, n_tail + 1, device=device).view(1, -1)
    target_tail = (base + offsets) % (vocab_size - 1)

    losses = []
    for _ in range(steps):
        loss = tail.training_loss(
            prefix_hidden,
            prefix_tokens,
            target_tail,
            ar.tok_emb,
            ar.tok_pos_emb,
            refine_iters=refine_iters,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    tail.eval()
    with torch.inference_mode():
        logits = tail(prefix_hidden, prefix_tokens, ar.tok_emb, ar.tok_pos_emb)
        pred = logits.argmax(dim=-1)
        acc = float((pred == target_tail).float().mean().item())

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1e-8),
        "tail_token_accuracy": acc,
    }


def run_reference_loss_consistency(
    vocab_size: int,
    d_model: int,
    prefix_len: int,
    n_tail: int,
    batch_size: int,
    device: torch.device,
) -> Dict:
    ar = _tiny_ar_model(vocab_size=vocab_size, d_model=d_model, max_seq=prefix_len + n_tail + 1).to(device)
    tail = ParallelTailDecoder(vocab_size=vocab_size, d_model=d_model, n_tail=n_tail, n_layers=2).to(device)
    tail.eval()

    prefix_hidden = torch.randn(batch_size, d_model, device=device)
    prefix_tokens = torch.randint(0, vocab_size - 1, (batch_size, prefix_len), device=device)
    target_tail = torch.randint(0, vocab_size - 1, (batch_size, n_tail), device=device)

    with torch.inference_mode():
        training_loss = tail.training_loss(
            prefix_hidden, prefix_tokens, target_tail, ar.tok_emb, ar.tok_pos_emb, refine_iters=1
        )
        logits = tail(prefix_hidden, prefix_tokens, ar.tok_emb, ar.tok_pos_emb)
        manual_loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_tail.reshape(-1))
        diff = float(torch.abs(training_loss - manual_loss).item())

    return {
        "training_loss": float(training_loss.item()),
        "manual_loss": float(manual_loss.item()),
        "abs_diff": diff,
    }


def run_refine_stability(
    vocab_size: int,
    d_model: int,
    prefix_len: int,
    n_tail: int,
    batch_size: int,
    device: torch.device,
) -> Dict:
    ar = _tiny_ar_model(vocab_size=vocab_size, d_model=d_model, max_seq=prefix_len + n_tail + 1).to(device)
    tail = ParallelTailDecoder(vocab_size=vocab_size, d_model=d_model, n_tail=n_tail, n_layers=2).to(device)
    tail.eval()

    prefix_hidden = torch.randn(batch_size, d_model, device=device)
    prefix_tokens = torch.randint(0, vocab_size - 1, (batch_size, prefix_len), device=device)

    with torch.inference_mode():
        logits_1 = tail(prefix_hidden, prefix_tokens, ar.tok_emb, ar.tok_pos_emb)
        ids_1 = logits_1.argmax(dim=-1)
        logits_2 = tail(prefix_hidden, prefix_tokens, ar.tok_emb, ar.tok_pos_emb, tail_token_ids=ids_1)
        ids_2 = logits_2.argmax(dim=-1)

        changed = float((ids_1 != ids_2).float().mean().item())
        max_logit_delta = float((logits_2 - logits_1).abs().max().item())
        finite = bool(torch.isfinite(logits_1).all() and torch.isfinite(logits_2).all())
        in_bounds = bool(
            (ids_1 >= 0).all() and (ids_1 < vocab_size).all() and (ids_2 >= 0).all() and (ids_2 < vocab_size).all()
        )

    return {
        "changed_token_fraction_iter2_vs_iter1": changed,
        "max_logit_delta": max_logit_delta,
        "finite_logits": finite,
        "token_bounds_ok": in_bounds,
    }


def run_real_batch_sanity(
    policy_checkpoint: str,
    prefix_len: int,
    refine_iters: int,
    device: torch.device,
) -> Dict:
    policy, cfg = BasePolicy.from_checkpoint(policy_checkpoint, return_configuration=True)
    policy.to(device)
    policy.eval()
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    batch = dataset[0]
    obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device)) for k, v in batch["obs"].items()}
    action = batch["action"]
    if not isinstance(action, torch.Tensor):
        action = torch.from_numpy(action)
    action = action.to(device).unsqueeze(0)
    obs = {k: v.unsqueeze(0) if v.dim() >= 1 else v for k, v in obs.items()}

    with torch.no_grad():
        target_tokens = policy.action_tokenizer.tokenize(action)
        cond, memory_is_embedded = policy._get_conditioning(obs)
        prefix_in = torch.full((1, 1), policy.bos_id, dtype=torch.long, device=device)
        _, prefix_hidden = policy.model.generate_prefix(
            prefix_in, cond, n_prefix_tokens=prefix_len, memory_is_embedded=memory_is_embedded, temperature=0.0
        )
        prefix_tokens = target_tokens[:, :prefix_len]
        target_tail = target_tokens[:, prefix_len:policy.max_seq_len]

        tail = policy.build_blockwise_tail_decoder(prefix_len=prefix_len).to(device)
        loss = tail.training_loss(
            prefix_hidden,
            prefix_tokens,
            target_tail,
            policy.model.tok_emb,
            policy.model.tok_pos_emb,
            refine_iters=refine_iters,
        )
    return {
        "loss": float(loss.item()),
        "finite_loss": bool(torch.isfinite(loss).all()),
        "target_tail_len": int(target_tail.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Blockwise-OAT training behavior")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--n-tail", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overfit-steps", type=int, default=120)
    parser.add_argument("--overfit-lr", type=float, default=3e-3)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--policy-checkpoint", default=None, help="Optional real-data one-batch sanity check")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail on weak but non-fatal diagnostics")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)

    overfit = run_micro_overfit(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        prefix_len=args.prefix_len,
        n_tail=args.n_tail,
        batch_size=args.batch_size,
        steps=args.overfit_steps,
        lr=args.overfit_lr,
        device=device,
        refine_iters=args.refine_iters,
    )
    reference = run_reference_loss_consistency(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        prefix_len=args.prefix_len,
        n_tail=args.n_tail,
        batch_size=max(4, args.batch_size // 2),
        device=device,
    )
    refine = run_refine_stability(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        prefix_len=args.prefix_len,
        n_tail=args.n_tail,
        batch_size=max(4, args.batch_size // 2),
        device=device,
    )

    real_batch = None
    if args.policy_checkpoint:
        real_batch = run_real_batch_sanity(
            policy_checkpoint=args.policy_checkpoint,
            prefix_len=args.prefix_len,
            refine_iters=args.refine_iters,
            device=device,
        )

    checks = {
        "overfit_ok": overfit["tail_token_accuracy"] > 0.99 and overfit["loss_ratio"] < 0.15,
        "reference_loss_ok": reference["abs_diff"] < 1e-6,
        "refine_numerics_ok": refine["finite_logits"] and refine["token_bounds_ok"],
    }
    if real_batch is not None:
        checks["real_batch_ok"] = real_batch["finite_loss"]

    passed = all(checks.values())
    if args.strict:
        passed = passed and refine["changed_token_fraction_iter2_vs_iter1"] > 0.0

    report = {
        "config": {
            "prefix_len": args.prefix_len,
            "n_tail": args.n_tail,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "overfit_steps": args.overfit_steps,
            "refine_iters": args.refine_iters,
            "device": str(device),
        },
        "overfit": overfit,
        "reference_loss": reference,
        "refine_stability": refine,
        "real_batch_sanity": real_batch,
        "checks": checks,
        "passed": passed,
    }

    out = json.dumps(report, indent=2, sort_keys=True)
    print(out)
    if args.report_json:
        report_path = pathlib.Path(args.report_json)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(out + "\n", encoding="utf-8")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
