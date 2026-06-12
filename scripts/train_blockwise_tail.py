#!/usr/bin/env python3
"""
Train ParallelTailDecoder on teacher action tokens from a frozen OAT policy.

Usage:
  python scripts/train_blockwise_tail.py \\
    --policy-checkpoint path/to/policy.ckpt \\
    --prefix-len 4 \\
    --epochs 10 \\
    --batch-size 64
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oat.blockwise_oat import ParallelTailDecoder
from oat.policy.base_policy import BasePolicy


def main():
    parser = argparse.ArgumentParser(description="Train blockwise OAT tail decoder")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="output/blockwise_tail_decoder.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, cfg = BasePolicy.from_checkpoint(
        args.policy_checkpoint, return_configuration=True,
    )
    assert hasattr(policy, "build_blockwise_tail_decoder"), "Checkpoint must be OATPolicy"
    policy = policy.to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    tail_decoder = policy.build_blockwise_tail_decoder(prefix_len=args.prefix_len).to(device)
    optimizer = torch.optim.AdamW(tail_decoder.parameters(), lr=args.lr)

    # Dataset from training config (Hydra-instantiated zarr loader).
    import hydra
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    p = args.prefix_len
    n_total = policy.max_seq_len
    assert p < n_total

    for epoch in range(args.epochs):
        tail_decoder.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if "obs" in batch:
                obs = {ok: ov.to(device) for ok, ov in batch["obs"].items()}
            else:
                obs = {ok: batch[ok].to(device) for ok in policy.obs_ports}

            with torch.no_grad():
                target_tokens = policy.action_tokenizer.tokenize(batch["action"].to(device))

            cond, memory_is_embedded = policy._get_conditioning(obs)
            b = cond.shape[0]
            prefix_in = torch.full((b, 1), policy.bos_id, dtype=torch.long, device=device)
            with torch.no_grad():
                _, prefix_hidden = policy.model.generate_prefix(
                    prefix_in, cond, n_prefix_tokens=p,
                    memory_is_embedded=memory_is_embedded,
                    temperature=0.0,
                )
            prefix_tokens = target_tokens[:, :p]
            target_tail = target_tokens[:, p:n_total]

            loss = tail_decoder.training_loss(
                prefix_hidden,
                prefix_tokens,
                target_tail,
                policy.model.tok_emb,
                policy.model.tok_pos_emb,
                refine_iters=args.refine_iters,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / max(n_batches, 1):.4f}")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prefix_len": args.prefix_len,
        "refine_iters": args.refine_iters,
        "state_dict": tail_decoder.state_dict(),
        "policy_checkpoint": str(args.policy_checkpoint),
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
