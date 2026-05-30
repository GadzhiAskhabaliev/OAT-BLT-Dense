---
name: Cross-Attn Dense OAT
overview: Внедрить dense visual memory для OAT-политики в OAT-BLT-Dense с таргетом обучения на LIBERO и сохранением fallback режима совместимости.
todos:
  - id: dense-encoder
    content: Добавить DenseRgbEncoder с выходом spatial tokens [B*To, L, d_model] в robomimic_vision_encoder.py, включая LayerNorm после 1x1-проекции
    status: pending
  - id: policy-memory
    content: Добавить use_cross_attn ветку в OATPolicy и реализовать get_dense_memory(obs_dict) с camera/time embeddings и state memory-токенами
    status: pending
  - id: decoder-alignment
    content: Адаптировать transformer_cache.AutoregressiveModel под длинный memory, отдельные positional embeddings памяти и pre-embedded cond без поломки legacy пути
    status: pending
  - id: hydra-config
    content: Протянуть use_cross_attn/dense_feature_dim/max_memory_len и параметры state-tokenization в train_oatpolicy.yaml и LIBERO task-конфиги
    status: pending
  - id: ckpt-compat
    content: Реализовать безопасную стратегию загрузки старых checkpoint (partial/strict=False) с явным логом missing/unexpected keys
    status: pending
  - id: attn-viz
    content: Добавить утилиту visualize_cross_attention с восстановлением (camera,time,h,w) и overlay heatmap
    status: pending
  - id: validation
    content: Провести smoke-валидацию train/infer и проверить корректность cross-attn KV-cache usage
    status: pending
---

# План внедрения Cross-Attention в BLT-OAT для LIBERO

## Контекст и ключевое решение
Целевой кодбейс: этот репозиторий (OAT-BLT-Dense). Пайплайн уже ориентирован на LIBERO (см. [`oat/config/train_oatpolicy.yaml`](../../oat/config/train_oatpolicy.yaml), `defaults: task/policy: libero/libero10`).

В этом репозитории cross-attention в декодере уже реализован в [`oat/model/autoregressive/transformer_cache.py`](../../oat/model/autoregressive/transformer_cache.py), включая одноразовый precompute `memory` K/V в `generate()`. Основная задача — перейти с pooled visual cond на dense spatial memory для LIBERO-потока.

## Целевые изменения по модулям
- **Dense RGB encoder**: добавить `DenseRgbEncoder` в [`oat/perception/robomimic_vision_encoder.py`](../../oat/perception/robomimic_vision_encoder.py).
  - Вход: `[B*To, 3, H, W]`.
  - Выход: `[B*To, L, d_model]`, где `L = h*w`.
  - Бэкбон: текущий resnet-путь до пространственной карты (без global pooling/SpatialSoftmax), плюс `1x1 Conv` проекция в `d_model`.
  - Добавить `LayerNorm(d_model)` после проекции и flatten (по последней размерности токена) для стабильности.
  - Для LIBERO (`128x128`) зафиксировать ожидание: после `layer3` при суммарном downsample `x16` получаем `8x8`, т.е. `L=64` токена на камеру; при `To=2` и 2 камерах `N_visual=256`.
  - Проверить входы LIBERO (`128x128`, 2 камеры) из [`oat/config/task/policy/libero/libero10.yaml`](../../oat/config/task/policy/libero/libero10.yaml) и зафиксировать ожидаемые `h,w,L`.
  - Проверить, что `d_model` dense-проекции точно совпадает с размерностью action-token embedding (`embed_dim` в policy/model).

- **Policy memory assembly**: модифицировать [`oat/policy/oatpolicy.py`](../../oat/policy/oatpolicy.py).
  - Добавить флаги: `use_cross_attn`, `dense_feature_dim`, `compress_visual_tokens`.
  - Реализовать `get_dense_memory(obs_dict)`:
    - отдельная/общая dense-обработка `agentview_rgb` и `robot0_eye_in_hand_rgb` (LIBERO);
    - добавление `time_embed` и `camera_embed` к каждому пространственному токену;
    - объединение в `memory: [B, N_total, d_model]`.
  - Для LIBERO state-части (`robot0_*`, `task_uid`) использовать единый путь через memory-токены:
    - `task_uid` через embedding;
    - конкатенация state + `task_uid_embed` и MLP-проекция в 1..K state-токенов;
    - конкатенация state-токенов к visual memory (без отдельной cond-ветки).
  - В `forward` (teacher forcing) и `predict_action` (autoregressive) передавать `memory` в `AutoregressiveModel` вместо текущего плоского `[B, To, D]` пути при `use_cross_attn=True`.

- **Decoder interface alignment**: точечно обновить [`oat/model/autoregressive/transformer_cache.py`](../../oat/model/autoregressive/transformer_cache.py) для dense memory.
  - Сохранить текущий блок `self-attn -> cross-attn -> ffn`.
  - Добавить явный режим «cond уже embedded memory» (чтобы не ломать legacy `cond_emb`).
  - Ввести отдельные positional embeddings для memory (`max_memory_len` с запасом, например 1024), не завязанные на `n_obs_steps`.
  - Проверить `max_cond_len`/позиционные эмбеддинги legacy-режима и развести их с memory-позицией, чтобы длинный `N_total` не ломал старый путь.
  - Не ломать текущий precompute `memory_kv_cache` в `generate()`.
  - Добавить проверку, что cross-attn KV-кеш вычисляется один раз перед decode loop и корректно индексируется по слоям при `use_cross_attn=True`.

- **Config wiring**: обновить [`oat/config/train_oatpolicy.yaml`](../../oat/config/train_oatpolicy.yaml) и LIBERO task-конфиги в [`oat/config/task/policy/libero/`](../../oat/config/task/policy/libero/).
  - Добавить новые флаги с дефолтами.
  - Добавить параметры `max_memory_len`, `num_state_tokens` (или эквивалент), и настройки `task_uid` embedding.
  - Оставить legacy режим (`use_cross_attn=False`) как совместимый fallback.
  - Явно валидировать совместимость с `task_uid` в shape_meta.

- **Checkpoint compatibility**: в [`oat/workspace/base_workspace.py`](../../oat/workspace/base_workspace.py) или policy-level load path предусмотреть частичную загрузку для старых весов.
  - Новые слои (`DenseRgbEncoder`, time/camera embeds, возможно новые pos embeddings) инициализировать случайно.
  - Документировать сценарий finetune: старый ckpt + `strict=False`.
  - Логировать `missing_keys`/`unexpected_keys` при загрузке, чтобы явно видеть отсутствие конфликтов по новым именам (`dense_encoder_*`, `*_embed`, `memory_pos_*`).

- **Attention visualization utility**: добавить новую утилиту (например, `oat/common/attention_viz.py` или `scripts/visualize_cross_attention.py`).
  - Собирать cross-attn weights по выбранному action token и усреднять по головам.
  - При формировании memory сохранять индексный mapping-буфер `(camera_idx, time_step, h_pos, w_pos, token_type)` для декодирования индексов.
  - Рендер heatmap overlay поверх RGB + отдельный вывод распределения внимания по `token_type` (visual/state).

## Поток данных (после внедрения)
```mermaid
flowchart TD
  obsRgb[Obs RGB BToHWC] --> denseEnc[DenseRgbEncoder]
  denseEnc --> camTimeEmb[Add camera and time embeddings]
  camTimeEmb --> memoryCat[Concat into memory BNtotalD]
  actionTok[Action token embeddings] --> decoder[AR Decoder blocks]
  memoryCat --> decoder
  decoder --> logits[Action token logits]
```

## Пошаговый порядок реализации (LIBERO-first)
1. Ввести `DenseRgbEncoder` и юнит-проверку форм тензоров.
2. Добавить `get_dense_memory()` и флаговый branch в `OATPolicy`, включая state как memory-токены.
3. Поддержать в `AutoregressiveModel` длинный `memory`, отдельные memory positional embeddings и режим pre-embedded cond.
4. Протянуть параметры в Hydra-конфиги с дефолтным таргетом LIBERO (`libero/libero10`) и включить `use_cross_attn=true` для нового эксперимента.
5. Добавить checkpoint migration/partial load стратегию.
6. Добавить attention-визуализацию для отладки.
7. Запустить smoke-тесты train/infer на 1 батче и коротком rollout через `LiberoRunner`.

## Проверки после внедрения
- Тренировка: `forward` с teacher forcing проходит без shape errors.
- Инференс: `generate()` использует одноразовый cross-attn KV precompute и не пересчитывает K/V памяти по шагам.
- Совместимость: старый checkpoint загружается в legacy режиме; в новом режиме корректно дообучается.
- Визуализация: для выбранного `token_idx` строится тепловая карта и корректно привязана к камере/времени.
- LIBERO: end-to-end eval через [`oat/env_runner/libero_runner.py`](../../oat/env_runner/libero_runner.py) проходит без деградации пайплайна данных.
- Производительность: проверить, что рост времени шага остаётся в рамках ожиданий для `N_memory≈256` и не становится bottleneck.
