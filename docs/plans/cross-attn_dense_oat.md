# Dense visual memory для OAT-BLT-Dense (LIBERO)

## Контекст

Целевой кодбейс: `OAT-BLT-Dense`. Пайплайн: [`oat/config/train_oatpolicy.yaml`](../../oat/config/train_oatpolicy.yaml), `task/policy: libero/libero10`.

**Принцип:** архитектуру AR-декодера не переписываем. Cross-attention уже реализован в [`oat/model/autoregressive/transformer_cache.py`](../../oat/model/autoregressive/transformer_cache.py). Меняется только тензор `memory` (perception + policy) и позиционные эмбеддинги под длину memory.

### Сжатая memory (baseline OAT)

```
RGB → pool → вектор на timestep; memory ≈ n_obs_steps токенов
```

### Dense memory (наша ветка)

```
RGB → feature map → patch-токены; memory = камеры × время × patches + state-токены
```

## Реализованные компоненты

| Компонент | Файл |
|-----------|------|
| `DenseRgbEncoder` | `oat/perception/robomimic_vision_encoder.py` |
| `get_dense_memory`, флаги абляций | `oat/policy/oatpolicy.py` |
| `memory_pos_emb`, `memory_is_embedded` | `oat/model/autoregressive/transformer_cache.py` |
| Hydra-конфиг | `oat/config/train_oatpolicy.yaml` |
| Визуализация cross-attn | `oat/common/attention_viz.py` |
| Загрузка legacy ckpt (`strict=False`) | `oat/policy/base_policy.py` |

## Флаги

- `policy.use_dense_visual_memory` (алиас: `use_cross_attn`)
- `policy.use_state_memory_tokens`
- `policy.use_task_uid_in_state_tokens`
- `policy.dense_feature_dim` / `embed_dim` (должны совпадать)
- `policy.max_memory_len`

## Валидация

1. Smoke: forward/backward, shapes, KV-cache при `memory_is_embedded=True`.
2. Short A/B на `libero10_N32`: сравнение с pooled baseline.
3. Long-run N500 + ladder SR 300/500/700.

Журнал эксперимента: [`docs/experiment_log_dense_visual_memory.md`](../experiment_log_dense_visual_memory.md).
