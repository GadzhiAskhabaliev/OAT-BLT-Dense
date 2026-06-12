# Blockwise OAT

Accelerated action-token generation for OAT policies: **P prefix tokens** are generated
autoregressively (prefix-decodable), **N = 8 − P tail tokens** in one parallel forward
pass via `ParallelTailDecoder`.

## Architecture

| Component | Role |
|-----------|------|
| `AutoregressiveModel.generate_prefix` | AR steps for z1..zP; returns `prefix_hidden` |
| `ParallelTailDecoder` | Mid-size transformer (default ~half AR depth, min ratio guard), bidirectional tail self-attn + cross-attn to prefix |
| `OATTok.detokenize` | Unchanged; full z1..z8 → action chunk |

### Trade-off (bidirectional tail)

Default tail self-attention is **bidirectional** for maximum speed in a single pass.
Prefixes of length **≤ P** remain prefix-decodable (AR prefix only).
Prefixes **P+1 .. 7** that include partial tail tokens are **not** guaranteed valid.

For strict decodability on all prefixes, use causal tail + iterative refinement (`refine_iters > 1`).

## Usage

```python
from oat.blockwise_oat import BlockwiseGenerateConfig, generate_with_blockwise_oat

tail = policy.build_blockwise_tail_decoder(prefix_len=4)
# ... load tail state_dict ...

cfg = BlockwiseGenerateConfig(prefix_len=4, total_tokens=8, refine_iters=1)
tokens, info = generate_with_blockwise_oat(
    policy.model, tail, cond, policy.bos_id, cfg,
    memory_is_embedded=policy.use_dense_visual_memory,
)
action = policy.action_tokenizer.detokenize(tokens)
```

Inference on policy:

```python
policy.blockwise_tail_decoder = tail
policy.use_blockwise_inference = True
out = policy.predict_action(obs_dict)  # or use_blockwise=True
```

## Training tail decoder

```bash
python scripts/train_blockwise_tail.py \
  --policy-checkpoint path/to/policy.ckpt \
  --prefix-len 4 \
  --refine-iters 1 \
  --epochs 10 \
  --output output/blockwise_tail_decoder.pt
```

Main OAT transformer is frozen; loss is CE on tail positions only.
By default, `build_blockwise_tail_decoder` enforces `tail_params / ar_params >= 0.35`
to avoid undersized tail models.

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `prefix_len` (P) | 4 | AR tokens z1..zP |
| `refine_iters` (k) | 1 | Tail re-feed iterations |
| `n_parallel_envs` | — | Eval only (separate) |

## Tests

```bash
pytest tests/test_blockwise_oat.py -q
```

## Verification workflow

```bash
# 1) Dataset integrity + prefix semantic checks (+ optional smoke subset)
python scripts/verify_blockwise_dataset.py \
  --policy-checkpoint path/to/policy.ckpt \
  --prefix-len 4 \
  --prefix-samples 300 \
  --smoke-subset-dst data/libero/libero10_smoke.zarr \
  --smoke-episodes 32

# 2) Training-loop checks (micro-overfit, loss consistency, refine stability)
python scripts/verify_blockwise_training.py \
  --prefix-len 4 \
  --n-tail 4 \
  --refine-iters 1 \
  --overfit-steps 120

# 3) Policy integration + AR vs Blockwise speed benchmark
python scripts/verify_blockwise_policy_integration.py \
  --policy-checkpoint path/to/policy.ckpt \
  --tail-checkpoint output/blockwise_tail_decoder.pt \
  --prefix-len 4 \
  --refine-iters 1
```

## Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Transformer tail (default)** | Models cross-token dependencies | More params than MLP |
| **MLP per position** | Very fast | Ignores tail interactions |
| **Causal tail + k iters** | Better prefix validity | Slower than 1-pass bidirectional |
