from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


@torch.no_grad()
def _extract_cross_attention_weights(
    policy,
    sample: Dict,
    token_idx: int,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        weights: [B, n_head, T_tok, T_mem] raw cross-attention weights (softmax over memory).
        memory: [B, T_mem, d_model] encoded memory used by cross-attention.
    """
    model = policy.model
    model.eval()
    policy.eval()

    if "obs" not in sample:
        raise ValueError("sample must contain 'obs' for memory construction.")

    cond, memory_is_embedded = policy._get_conditioning(sample["obs"])
    memory = model._encode_memory(cond, memory_is_embedded=memory_is_embedded)

    if "action_tokens" in sample:
        action_tokens = sample["action_tokens"].long().to(memory.device)
    elif "action" in sample:
        # Teacher-forcing path: build decoder input from expert actions.
        action_tokens = policy.action_tokenizer.tokenize(sample["action"])
        bos = torch.full(
            (action_tokens.shape[0], 1),
            policy.bos_id,
            dtype=torch.long,
            device=action_tokens.device,
        )
        action_tokens = torch.cat([bos, action_tokens], dim=1)
    else:
        raise ValueError("sample must contain either 'action_tokens' or 'action'.")

    # Decoder input sequence (exclude target shift-right last token).
    x_tokens = action_tokens[:, :-1]
    bsz, t_tok = x_tokens.shape
    if token_idx < 0:
        token_idx = t_tok + token_idx
    if token_idx < 0 or token_idx >= t_tok:
        raise ValueError(f"token_idx={token_idx} out of range for T_tok={t_tok}")

    tok_emb = model.tok_emb(x_tokens)
    pos_emb = model.tok_pos_emb[:, :t_tok, :]
    x = model.drop(tok_emb + pos_emb)

    n_layer = len(model.blocks)
    if layer_idx < 0:
        layer_idx = n_layer + layer_idx
    if layer_idx < 0 or layer_idx >= n_layer:
        raise ValueError(f"layer_idx={layer_idx} out of range for n_layer={n_layer}")

    weights = None
    for i, block in enumerate(model.blocks):
        # Self-attention branch
        attn_out, _ = block.attn(block.ln_1(x), layer_past=None)
        x = x + attn_out

        q_in = block.ln_2(x)
        if i == layer_idx:
            q = block.cross_attn.q_proj(q_in)
            q = q.view(bsz, t_tok, model.n_head, model.n_emb // model.n_head).transpose(1, 2)
            k, _v = block.cross_attn.kv_proj(memory).split(model.n_emb, dim=2)
            k = k.view(memory.shape[0], memory.shape[1], model.n_head, model.n_emb // model.n_head).transpose(1, 2)
            # [B, H, T_tok, T_mem]
            logits = torch.matmul(q, k.transpose(-2, -1)) * (block.cross_attn.head_dim ** -0.5)
            weights = torch.softmax(logits, dim=-1)

        # Cross-attention + MLP branch
        cross_out = block.cross_attn(q_in, memory)
        x = x + cross_out
        x = x + block.mlp(block.ln_3(x))

    if weights is None:
        raise RuntimeError("Failed to extract cross-attention weights.")
    return weights, memory


@torch.no_grad()
def visualize_cross_attention(
    policy,
    sample: Dict,
    token_idx: int,
    batch_index: int = 0,
    layer_idx: int = -1,
    alpha: float = 0.45,
):
    """
    Visualize cross-attention heatmaps for one action token against dense memory.

    Requirements:
      - policy.use_dense_visual_memory=True
      - policy.get_dense_memory builds policy.memory_index_map with
        (camera_idx, time_step, h_pos, w_pos, token_type)

    Returns:
      fig, diagnostics dict
    """
    if not getattr(policy, "use_dense_visual_memory", False):
        raise ValueError("visualize_cross_attention requires use_dense_visual_memory=True")

    weights, _memory = _extract_cross_attention_weights(policy, sample, token_idx, layer_idx=layer_idx)
    # [H, T_mem] for selected batch/token
    token_weights = weights[batch_index, :, token_idx, :]
    mean_weights = token_weights.mean(dim=0)  # [T_mem]

    if policy.memory_index_map is None:
        raise ValueError("policy.memory_index_map is empty; run dense memory path before visualization.")
    index_map = policy.memory_index_map.to(mean_weights.device)
    if index_map.shape[0] != mean_weights.shape[0]:
        raise ValueError(
            f"index_map length {index_map.shape[0]} != memory length {mean_weights.shape[0]}"
        )

    rows = index_map.cpu()
    vals = mean_weights.detach().cpu()

    obs = sample["obs"]
    camera_keys = list(policy.rgb_camera_keys)
    n_cam = len(camera_keys)
    to = int(obs[camera_keys[0]].shape[1])

    fig, axes = plt.subplots(
        to, n_cam, figsize=(5 * n_cam, 4 * to), squeeze=False
    )

    # Visual token overlays
    for t in range(to):
        for c in range(n_cam):
            ax = axes[t][c]
            img = obs[camera_keys[c]][batch_index, t].detach().cpu()
            if img.shape[-1] != 3:
                raise ValueError(f"Expected HWC image for {camera_keys[c]}, got shape {tuple(img.shape)}")

            mask = (
                (rows[:, 0] == c)
                & (rows[:, 1] == t)
                & (rows[:, 4] == 0)
            )
            heat_rows = rows[mask]
            heat_vals = vals[mask]

            if heat_rows.numel() == 0:
                ax.imshow(img)
                ax.set_title(f"{camera_keys[c]} t={t} (no tokens)")
                ax.axis("off")
                continue

            h_max = int(heat_rows[:, 2].max().item()) + 1
            w_max = int(heat_rows[:, 3].max().item()) + 1
            heat = torch.zeros(h_max, w_max)
            for rr, vv in zip(heat_rows, heat_vals):
                heat[int(rr[2]), int(rr[3])] = float(vv)

            heat = heat / (heat.max() + 1e-8)
            heat = F.interpolate(
                heat[None, None],
                size=(img.shape[0], img.shape[1]),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            ax.imshow(img)
            ax.imshow(heat.numpy(), cmap="jet", alpha=alpha)
            ax.set_title(f"{camera_keys[c]} t={t}")
            ax.axis("off")

    # Add state-attention summary as a separate small figure.
    state_mask = rows[:, 4] == 1
    state_vals = vals[state_mask]
    diagnostics = {
        "mean_attention_per_memory_token": vals,
        "state_attention_total": float(state_vals.sum().item()) if state_vals.numel() else 0.0,
        "state_attention_mean": float(state_vals.mean().item()) if state_vals.numel() else 0.0,
        "token_idx": token_idx,
        "layer_idx": layer_idx,
    }

    fig.tight_layout()
    return fig, diagnostics

