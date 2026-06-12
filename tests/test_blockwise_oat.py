"""Unit tests for blockwise OAT parallel tail decoder."""

import torch

from oat.blockwise_oat import (
    BlockwiseGenerateConfig,
    ParallelTailDecoder,
    benchmark_blockwise_vs_ar,
    generate_with_blockwise_oat,
)
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


def _tiny_ar_model(vocab_size=32, d_model=64, max_seq=9):
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


class _IdentityTokenizer:
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        # Test double: any valid token tensor should decode without errors.
        assert tokens.ndim == 2
        return tokens.float()


def test_parallel_tail_decoder_output_shape():
    b, p, n_tail, d = 2, 4, 4, 64
    vocab = 32
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    tail = ParallelTailDecoder(vocab_size=vocab, d_model=d, n_tail=n_tail)

    prefix_hidden = torch.randn(b, d)
    prefix_tokens = torch.randint(0, vocab - 1, (b, p))
    logits = tail(prefix_hidden, prefix_tokens, ar.tok_emb, ar.tok_pos_emb)
    assert logits.shape == (b, n_tail, vocab)


def test_generate_with_blockwise_oat_full_sequence():
    b, d = 2, 64
    vocab = 32
    bos_id = vocab - 1
    total = 8
    p = 4
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    tail = ParallelTailDecoder(vocab_size=vocab, d_model=d, n_tail=total - p)
    cond = torch.randn(b, 2, d)

    tokens, info = generate_with_blockwise_oat(
        ar, tail, cond, bos_id,
        BlockwiseGenerateConfig(prefix_len=p, total_tokens=total, temperature=0.0, use_argmax_tail=True),
    )
    assert tokens.shape == (b, total)
    assert info["prefix_tokens"].shape == (b, p)
    assert info["tail_tokens"].shape == (b, total - p)


def test_prefix_hidden_matches_generate_prefix():
    b, d = 1, 64
    vocab = 32
    bos_id = vocab - 1
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    cond = torch.randn(b, 2, d)
    prefix = torch.full((b, 1), bos_id, dtype=torch.long)
    out, hidden = ar.generate_prefix(prefix, cond, n_prefix_tokens=4, temperature=0.0)
    assert out.shape == (b, 5)
    assert hidden.shape == (b, d)


def test_refine_iters_zero_runs_single_pass():
    b, d = 2, 64
    vocab = 32
    bos_id = vocab - 1
    total = 8
    p = 4
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    tail = ParallelTailDecoder(vocab_size=vocab, d_model=d, n_tail=total - p)
    cond = torch.randn(b, 2, d)

    tokens0, info0 = generate_with_blockwise_oat(
        ar,
        tail,
        cond,
        bos_id,
        BlockwiseGenerateConfig(prefix_len=p, total_tokens=total, refine_iters=0, temperature=0.0, use_argmax_tail=True),
    )
    tokens1, info1 = generate_with_blockwise_oat(
        ar,
        tail,
        cond,
        bos_id,
        BlockwiseGenerateConfig(prefix_len=p, total_tokens=total, refine_iters=1, temperature=0.0, use_argmax_tail=True),
    )
    assert tokens0.shape == (b, total)
    assert info0["tail_tokens"].shape == (b, total - p)
    assert info0["refine_iters_used"] == 1
    assert torch.equal(info0["prefix_tokens"], info1["prefix_tokens"])


def test_prefix_decodability():
    b, d = 1, 64
    vocab = 32
    bos_id = vocab - 1
    total = 8
    p = 4
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    tail = ParallelTailDecoder(vocab_size=vocab, d_model=d, n_tail=total - p)
    cond = torch.randn(b, 2, d)

    prefix = torch.full((b, 1), bos_id, dtype=torch.long)
    full_ar_with_bos = ar.generate(prefix, cond, max_new_tokens=total, temperature=0.0)
    expected_prefix = full_ar_with_bos[:, 1:1 + p]

    _, info = generate_with_blockwise_oat(
        ar,
        tail,
        cond,
        bos_id,
        BlockwiseGenerateConfig(prefix_len=p, total_tokens=total, temperature=0.0, use_argmax_tail=True),
    )

    assert torch.equal(info["prefix_tokens"], expected_prefix)
    tokenizer = _IdentityTokenizer()
    action_from_prefix = tokenizer.detokenize(info["prefix_tokens"])
    action_from_full_prefix = tokenizer.detokenize(expected_prefix)
    assert action_from_prefix is not None
    assert action_from_full_prefix is not None
    assert action_from_prefix.shape == action_from_full_prefix.shape


def test_benchmark_runs():
    b, d = 1, 64
    vocab = 32
    bos_id = vocab - 1
    ar = _tiny_ar_model(vocab_size=vocab, d_model=d)
    tail = ParallelTailDecoder(vocab_size=vocab, d_model=d, n_tail=4)
    cond = torch.randn(b, 2, d)
    stats = benchmark_blockwise_vs_ar(
        ar, tail, cond, bos_id, warmup=2, repeats=6,
    )
    assert stats["speedup"] > 1.0, f"Blockwise is slower than AR: {stats}"
    assert stats["blockwise_mean_sec"] > 0
