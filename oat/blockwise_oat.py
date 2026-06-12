"""
Blockwise OAT: AR prefix tokens + parallel tail generation.

First P action tokens (z1..zP) are generated autoregressively by the main OAT
decoder. Remaining N = total_tokens - P tokens are predicted in one (or k)
forward pass(es) by a lightweight ParallelTailDecoder.

Prefix-decodability is preserved for prefixes of length <= P (first P tokens
still come from the main AR model). Longer prefixes that include parallel tail
tokens are not guaranteed to decode to valid actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.model.autoregressive.transformer_cache import CrossAttention, MLP, RMSNorm


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """logits: [B, vocab] -> token ids [B, 1]"""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class BidirectionalSelfAttention(nn.Module):
    """Full (non-causal) self-attention for parallel tail refinement."""

    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        assert n_emb % n_head == 0
        self.c_attn = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_emb, dim=-1)
        q, k, v = map(
            lambda t_: t_.view(b, t, self.n_head, self.head_dim).transpose(1, 2),
            (q, k, v),
        )
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False,
            dropout_p=self.p_attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class TailDecoderBlock(nn.Module):
    """Bidirectional self-attn + cross-attn to prefix + FFN."""

    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.ln_1 = RMSNorm(n_emb)
        self.self_attn = BidirectionalSelfAttention(n_head, n_emb, p_drop_attn)
        self.ln_2 = RMSNorm(n_emb)
        self.cross_attn = CrossAttention(n_head, n_emb, p_drop_attn)
        self.ln_3 = RMSNorm(n_emb)
        self.mlp = MLP(n_emb, p_drop_attn)

    def forward(self, x: torch.Tensor, prefix_memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), prefix_memory)
        x = x + self.mlp(self.ln_3(x))
        return x


class ParallelTailDecoder(nn.Module):
    """
    Lightweight decoder for parallel prediction of tail action tokens.

    Uses bidirectional self-attention within the tail for one-shot prediction.
    Trade-off: faster than full AR, but prefixes z1..z_{P+k} for k>0 are not
    guaranteed prefix-decodable unless you use causal tail + iterative decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_tail: int,
        prefix_pos_offset: int = 1,
        n_layers: int = 2,
        n_heads: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_tail = n_tail
        self.prefix_pos_offset = prefix_pos_offset

        self.tail_pos_emb = nn.Parameter(torch.zeros(1, n_tail, d_model))
        nn.init.normal_(self.tail_pos_emb, std=0.02)
        self.prefix_cond_proj = nn.Linear(d_model, d_model, bias=False)

        self.blocks = nn.ModuleList([
            TailDecoderBlock(n_heads, d_model, p_drop) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _build_prefix_memory(
        self,
        prefix_hidden: torch.Tensor,
        prefix_tokens: torch.Tensor,
        token_embedding: nn.Embedding,
        main_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Memory for cross-attention: prefix token states + global cond."""
        b, p = prefix_tokens.shape
        tok_emb = token_embedding(prefix_tokens)
        # Positions in main AR stream: offset .. offset+P-1 (z1 at index 1 after BOS).
        pos = main_pos_emb[:, self.prefix_pos_offset:self.prefix_pos_offset + p, :]
        prefix_embeds = tok_emb + pos
        global_cond = self.prefix_cond_proj(prefix_hidden).unsqueeze(1)
        return torch.cat([prefix_embeds, global_cond], dim=1)

    def _tail_inputs(
        self,
        prefix_hidden: torch.Tensor,
        tail_token_ids: Optional[torch.Tensor],
        token_embedding: nn.Embedding,
    ) -> torch.Tensor:
        b = prefix_hidden.shape[0]
        if tail_token_ids is None:
            queries = self.tail_pos_emb.expand(b, -1, -1).clone()
        else:
            queries = token_embedding(tail_token_ids) + self.tail_pos_emb
        return queries + prefix_hidden.unsqueeze(1)

    def forward(
        self,
        prefix_hidden: torch.Tensor,
        prefix_tokens: torch.Tensor,
        token_embedding: nn.Embedding,
        main_pos_emb: torch.Tensor,
        tail_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prefix_hidden: [B, d_model] last hidden state of main AR after z_P.
            prefix_tokens: [B, P] indices z1..zP (no BOS).
            token_embedding: shared embedding table from main AR model.
            main_pos_emb: [1, max_seq, d_model] positional embeddings from main AR.
            tail_token_ids: optional [B, N] current tail guess for refinement.

        Returns:
            logits: [B, N, vocab_size]
        """
        prefix_memory = self._build_prefix_memory(
            prefix_hidden, prefix_tokens, token_embedding, main_pos_emb,
        )
        x = self._tail_inputs(prefix_hidden, tail_token_ids, token_embedding)
        for block in self.blocks:
            x = block(x, prefix_memory)
        x = self.ln_f(x)
        return self.head(x)

    def training_loss(
        self,
        prefix_hidden: torch.Tensor,
        prefix_tokens: torch.Tensor,
        target_tail_tokens: torch.Tensor,
        token_embedding: nn.Embedding,
        main_pos_emb: torch.Tensor,
        refine_iters: int = 1,
    ) -> torch.Tensor:
        """Cross-entropy on all tail positions (teacher forcing on prefix)."""
        n_iters = max(1, int(refine_iters))
        tail_ids = None
        loss = 0.0

        for it in range(n_iters):
            logits = self.forward(
                prefix_hidden,
                prefix_tokens,
                token_embedding,
                main_pos_emb,
                tail_token_ids=tail_ids,
            )
            loss = loss + F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_tail_tokens.reshape(-1),
            )
            if it < n_iters - 1:
                tail_ids = logits.argmax(dim=-1).detach()
        return loss / n_iters


@dataclass
class BlockwiseGenerateConfig:
    prefix_len: int = 4
    total_tokens: int = 8
    refine_iters: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = 10
    use_argmax_tail: bool = False


@torch.inference_mode()
def generate_with_blockwise_oat(
    ar_model,
    tail_decoder: ParallelTailDecoder,
    cond: torch.Tensor,
    bos_id: int,
    cfg: Optional[BlockwiseGenerateConfig] = None,
    memory_is_embedded: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Blockwise generation: AR prefix + parallel tail.

    Returns:
        full_tokens: [B, total_tokens] action token indices z1..z8 (no BOS).
        info: timing and intermediate tensors.
    """
    if cfg is None:
        cfg = BlockwiseGenerateConfig()
    p = cfg.prefix_len
    n_total = cfg.total_tokens
    n_tail = n_total - p
    refine_iters = max(1, int(cfg.refine_iters))
    assert 1 <= p < n_total

    b = cond.shape[0]
    device = cond.device
    prefix = torch.full((b, 1), bos_id, dtype=torch.long, device=device)

    t0 = time.perf_counter()
    out_with_bos, prefix_hidden = ar_model.generate_prefix(
        prefix,
        cond,
        n_prefix_tokens=p,
        memory_is_embedded=memory_is_embedded,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
    )
    t_ar = time.perf_counter() - t0

    prefix_tokens = out_with_bos[:, 1:1 + p]
    tail_ids = None
    t1 = time.perf_counter()
    for _ in range(refine_iters):
        logits = tail_decoder(
            prefix_hidden,
            prefix_tokens,
            ar_model.tok_emb,
            ar_model.tok_pos_emb,
            tail_token_ids=tail_ids,
        )
        if cfg.use_argmax_tail or cfg.temperature <= 0:
            tail_ids = logits.argmax(dim=-1)
        else:
            flat = logits.reshape(-1, logits.size(-1))
            sampled = _sample_logits(flat, cfg.temperature, cfg.top_k)
            tail_ids = sampled.view(b, n_tail)
        if (tail_ids < 0).any() or (tail_ids >= tail_decoder.vocab_size).any():
            raise ValueError(
                f"Generated tail token index out of bounds: "
                f"{tail_ids.min().item()}..{tail_ids.max().item()} vs vocab_size={tail_decoder.vocab_size}"
            )
    t_tail = time.perf_counter() - t1

    if tail_ids is None:
        raise RuntimeError("Tail generation produced no tokens. Check refine_iters and tail decoder setup.")

    full_tokens = torch.cat([prefix_tokens, tail_ids], dim=1)
    info = {
        "prefix_tokens": prefix_tokens,
        "tail_tokens": tail_ids,
        "prefix_hidden": prefix_hidden,
        "ar_seconds": t_ar,
        "tail_seconds": t_tail,
        "prefix_len": p,
        "tail_len": n_tail,
        "refine_iters_used": refine_iters,
        # Prefix-decodability guarantee for bidirectional tail:
        # only AR-generated z1..zP is strictly prefix-decodable.
        "prefix_decodable_upto": p,
        "tail_prefixes_prefix_decodable": False,
    }
    return full_tokens, info


def decode_prefix_action(action_tokenizer, prefix_tokens: torch.Tensor, p: int) -> torch.Tensor:
    """Decode first p tokens only (prefix-decodable by construction)."""
    return action_tokenizer.detokenize(prefix_tokens[:, :p])


@torch.inference_mode()
def benchmark_blockwise_vs_ar(
    ar_model,
    tail_decoder: ParallelTailDecoder,
    cond: torch.Tensor,
    bos_id: int,
    total_tokens: int = 8,
    prefix_len: int = 4,
    memory_is_embedded: bool = False,
    warmup: int = 2,
    repeats: int = 10,
) -> dict:
    """Compare wall-clock AR (total_tokens steps) vs blockwise (prefix_len AR + 1 tail)."""
    cfg = BlockwiseGenerateConfig(
        prefix_len=prefix_len,
        total_tokens=total_tokens,
        refine_iters=1,
        temperature=0.0,
        use_argmax_tail=True,
    )

    def run_ar():
        prefix = torch.full((cond.shape[0], 1), bos_id, dtype=torch.long, device=cond.device)
        return ar_model.generate(
            prefix, cond, max_new_tokens=total_tokens,
            memory_is_embedded=memory_is_embedded, temperature=0.0,
        )

    def run_blockwise():
        return generate_with_blockwise_oat(
            ar_model, tail_decoder, cond, bos_id, cfg, memory_is_embedded,
        )

    for _ in range(warmup):
        run_ar()
        run_blockwise()

    ar_times, bw_times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_ar()
        ar_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_blockwise()
        bw_times.append(time.perf_counter() - t0)

    import statistics
    return {
        "ar_mean_sec": statistics.mean(ar_times),
        "blockwise_mean_sec": statistics.mean(bw_times),
        "speedup": statistics.mean(ar_times) / max(statistics.mean(bw_times), 1e-9),
        "prefix_len": prefix_len,
        "total_tokens": total_tokens,
    }
