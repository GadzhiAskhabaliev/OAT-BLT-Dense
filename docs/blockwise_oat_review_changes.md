# Blockwise-OAT: описание изменений для научного ревью

**Ветка:** `Blockwise-OAT`  
**Базовая линия:** `main` (BLT-OAT-Dense — dense visual memory + стандартный AR-декодер OAT)  
**Дата документа:** 2026-06-12  
**Статус:** архитектура реализована и покрыта unit/verification-тестами; обучение tail-декодера на полном LIBERO и sim-eval Blockwise vs AR — **ещё не завершены**.

---

## 1. Мотивация

Стандартный OAT генерирует 8 action-токенов (`z1..z8`) **полностью авторегрессионно** (8 последовательных forward-pass'ов декодера). Это узкое место по латентности на инференсе.

**Blockwise-OAT** разделяет генерацию на два этапа:

1. **Префикс `z1..zP`** — как в исходном OAT, строго AR (prefix-decodable).
2. **Хвост `z_{P+1}..z8`** — одним (или несколькими) forward-pass'ом лёгкого модуля `ParallelTailDecoder`.

Полная последовательность `z1..z8` по-прежнему подаётся в **неизменённый** `OATTok.detokenize` → совместимость с существующим токенизатором/детокенизатором сохранена.

**Целевой выигрыш:** уменьшение числа AR-шагов с 8 до `P` + 1 (при `refine_iters=1`), при настраиваемом `P` (по умолчанию 4).

---

## 2. Архитектура (детально)

### 2.1. Префикс: исходный AR-декодер без изменений логики

Добавлен метод `AutoregressiveModel.generate_prefix()` (`oat/model/autoregressive/transformer_cache.py`):

- Авторегрессивно генерирует ровно `P` токенов после BOS.
- Использует тот же KV-cache, embedding table и positional embeddings, что и `generate()`.
- Возвращает:
  - `out_tokens` — последовательность с BOS;
  - `prefix_hidden` — скрытое состояние декодера после `z_P` (для conditioning хвоста).

**Свойство prefix-decodability:** префикс длины `≤ P` состоит только из AR-токенов основного декодера и может быть детокенизирован отдельно (валидное частичное действие в смысле OAT).

### 2.2. Хвост: `ParallelTailDecoder`

Файл: `oat/blockwise_oat.py`

| Параметр | Значение по умолчанию |
|----------|----------------------|
| Число слоёв | 2 |
| Число голов | 4 |
| Self-attention внутри хвоста | **Bidirectional** (non-causal) |
| Cross-attention | К памяти префикса |
| Выход | `logits [B, N_tail, vocab_size]` |

**Conditioning от префикса** (два канала):

1. **Token-level:** эмбеддинги `z1..zP` + позиционные embeddings основного AR (`main_pos_emb`, смещение 1 после BOS).
2. **Global:** `prefix_hidden` — последний hidden state AR после `z_P`; проецируется и добавляется в cross-attention memory и в query-вход хвоста.

**Trade-off (явно задокументирован):** при bidirectional tail префиксы длины `P+1..7`, включающие частично сгенерированный хвост, **не гарантированно** валидны для детокенизатора. Это осознанный компромисс в пользу скорости одного прохода.

### 2.3. Сборка и детокенизация

`generate_with_blockwise_oat()`:

```
full_tokens = cat([prefix_tokens, tail_ids], dim=1)   # [B, 8]
action = OATTok.detokenize(full_tokens)
```

Детокенизатор не знает о blockwise-генерации.

### 2.4. Настраиваемые параметры

| Параметр | Где | Смысл |
|----------|-----|-------|
| `prefix_len` (P) | `BlockwiseGenerateConfig`, `OATPolicy` | Длина AR-префикса |
| `refine_iters` (k) | config / training | Число итераций уточнения хвоста (re-feed `tail_ids`) |
| `total_tokens` | config | Обычно 8 (`latent_horizon`) |
| `use_blockwise_inference` | `OATPolicy` | Включить blockwise в `predict_action` |

---

## 3. Интеграция в `OATPolicy`

Файл: `oat/policy/oatpolicy.py`

Новые поля конструктора:

- `use_blockwise_inference` (default `False`)
- `blockwise_prefix_len` (default `4`)
- `blockwise_refine_iters` (default `1`)
- `blockwise_tail_decoder` (optional `ParallelTailDecoder`)

Новые методы:

- `predict_action_blockwise()` — AR-префикс + parallel tail + `detokenize`
- `build_blockwise_tail_decoder()` — фабрика tail-модуля под размер policy

`predict_action(..., use_blockwise=True)` делегирует в blockwise-путь.

**Fallback:** если `use_blockwise=True`, но `blockwise_tail_decoder` не загружен — **warning + автоматический переход на full AR** (не падаем с `RuntimeError`).

---

## 4. Обучение tail-декодера

Скрипт: `scripts/train_blockwise_tail.py`

- Загружает frozen `OATPolicy` из checkpoint.
- Основной AR-трансформер и action tokenizer **заморожены**.
- Обучает только `ParallelTailDecoder` по CE на позициях хвоста.
- Teacher forcing: `prefix_hidden` из `generate_prefix`, `prefix_tokens` и `target_tail` из `action_tokenizer.tokenize(action)`.
- Сохраняет `.pt` с `state_dict`, `prefix_len`, `refine_iters`.

**Статус:** скрипт готов; полноценный прогон на LIBERO-10 (500 demo) на кластере **ещё не выполнен**.

---

## 5. Исправления после архитектурного аудита

После code review выявлены и исправлены следующие проблемы:

### 5.1. `refine_iters=0` ломал генерацию

**Было:** при `refine_iters=0` цикл не выполнялся, `tail_ids=None` → падение в `torch.cat`.

**Стало:** `refine_iters = max(1, refine_iters)`; в `info` пишется `refine_iters_used`.

### 5.2. `training_loss` не обновлял `tail_ids` между итерациями

**Было:** `tail_ids = argmax(...)` вычислялся один раз; все refinement-итерации использовали один и тот же вход.

**Стало:** на каждой итерации `tail_ids = logits.argmax(dim=-1).detach()` перед следующим forward (кроме последней).

### 5.3. Проверка границ словаря

Добавлены `ValueError` при выходе индексов за `[0, vocab_size)`:

- после генерации tail в `generate_with_blockwise_oat`;
- после каждого AR-шага в `generate_prefix`.

### 5.4. Fallback при отсутствии tail-декодера

`predict_action_blockwise` при `blockwise_tail_decoder is None` → warning + `predict_action(..., use_blockwise=False)`.

---

## 6. Verification-kit (трёхуровневая проверка)

Добавлены скрипты для воспроизводимой валидации **до** полного обучения и sim-eval:

### Уровень 1 — данные (`scripts/verify_blockwise_dataset.py`)

- Структурная целостность zarr-датасета (ключи obs, формы, finite).
- Семантика префикса: `tokenize(action)` → `detokenize(z1..zP)` без NaN/Inf.
- Опционально: создание smoke-subset `.zarr` для быстрых итераций.

### Уровень 2 — обучение (`scripts/verify_blockwise_training.py`)

- **Micro-overfit** на синтетике (детерминированный mapping prefix→tail); критерий: accuracy → 100%.
- **Reference loss:** `training_loss` vs ручной CE на одном forward — должны совпадать (`abs_diff < 1e-6`).
- **Refinement stability:** токены/логиты меняются между итерациями 1 и 2, без расходимости.
- Опционально: one-batch sanity на реальном checkpoint.

### Уровень 3 — policy (`scripts/verify_blockwise_policy_integration.py`)

- Совпадение форматов выходов AR vs Blockwise (`action`, `action_pred`).
- Проверка fallback без tail-декодера.
- Бенчмарк `benchmark_blockwise_vs_ar` — критерий `speedup > 1`.

Документация по запуску: `docs/blockwise_oat.md` (раздел *Verification workflow*).

---

## 7. Тесты

Файл: `tests/test_blockwise_oat.py` — **6 тестов**, все проходят локально:

| Тест | Что проверяет |
|------|----------------|
| `test_parallel_tail_decoder_output_shape` | Форма logits `[B, N, vocab]` |
| `test_generate_with_blockwise_oat_full_sequence` | Полная последовательность длины 8 |
| `test_prefix_hidden_matches_generate_prefix` | API `generate_prefix` |
| `test_refine_iters_zero_runs_single_pass` | `refine_iters=0` не падает |
| `test_prefix_decodability` | Blockwise-префикс = AR-префикс при тех же cond |
| `test_benchmark_runs` | `speedup > 1.0` на micro-benchmark |

Запуск: `pytest tests/test_blockwise_oat.py -q`

---

## 8. Полный список изменённых/добавленных файлов

### Новые файлы

| Файл | Назначение |
|------|------------|
| `oat/blockwise_oat.py` | `ParallelTailDecoder`, `generate_with_blockwise_oat`, benchmark |
| `docs/blockwise_oat.md` | Краткая документация и usage |
| `docs/blockwise_oat_review_changes.md` | Этот документ |
| `scripts/train_blockwise_tail.py` | Обучение tail на frozen policy |
| `scripts/verify_blockwise_dataset.py` | Верификация датасета |
| `scripts/verify_blockwise_training.py` | Верификация training loop |
| `scripts/verify_blockwise_policy_integration.py` | Интеграция + speedup |
| `tests/test_blockwise_oat.py` | Unit-тесты |

### Изменённые файлы (относительно `main`)

| Файл | Изменение |
|------|-----------|
| `oat/model/autoregressive/transformer_cache.py` | `generate_prefix()` + vocab bounds check |
| `oat/policy/oatpolicy.py` | blockwise inference API, fallback, `build_blockwise_tail_decoder` |
| `scripts/eval_policy_sim.py` | флаги `--use-blockwise`, `--blockwise-prefix-len`, `--blockwise-refine-iters` (для будущего sim-eval) |

### Что **не** входит в эту ветку

Ветка `main` содержит отдельную линию работы **BLT-OAT-Dense**:

- результаты обучения dense policy (ladder 300/500/700/950);
- Phase B confirm eval;
- HF publish, dashboard scripts.

Эти изменения **не смешиваются** с `Blockwise-OAT` до явного merge/cherry-pick.

---

## 9. Результаты архитектурного аудита (чек-лист)

| Требование | Вердикт |
|------------|---------|
| AR-префикс через исходный декодер | ✅ `generate_prefix` |
| Префикс не перезаписывается при генерации хвоста | ✅ `cat(prefix, tail)` |
| Prefix-decodability для `z1..zP` | ✅ по построению; тест на совпадение с AR-префиксом |
| Parallel tail — lightweight transformer | ✅ 2 слоя, bidirectional + cross-attn |
| Conditioning: hidden + token embeddings | ✅ `_build_prefix_memory`, `_tail_inputs` |
| `OATTok.detokenize` без изменений | ✅ |
| Настраиваемые `P`, `refine_iters` | ✅ |
| Speedup > 1 (micro-benchmark) | ✅ тест + `benchmark_blockwise_vs_ar` |
| Fallback на full AR | ✅ (после исправления) |
| Обучение с `refine_iters > 1` | ✅ исправлено обновление `tail_ids` |

**Ограничение:** префиксы длины `> P` с частичным хвостом могут быть невалидны (by design при bidirectional tail).

---

## 10. План следующих шагов (для полной научной валидации)

1. **Обучить** `ParallelTailDecoder` на LIBERO-10 от лучшего dense checkpoint (`ep-0950` или актуального baseline).
2. **Запустить verification-kit** (шаги 1→2→3) на кластере с реальным checkpoint и smoke/full dataset.
3. **Sim-eval LIBERO-10:** сравнить SR Blockwise vs full AR (`eval_policy_sim.py --use-blockwise`); критерий приемлемости: падение SR ≤ 2–3% от AR baseline при `speedup > 1`.
4. При необходимости: увеличить `P`, `refine_iters`, или число эпох обучения tail.
5. После валидации — PR `Blockwise-OAT` → `main` (точечный merge, без dense-артефактов).

---

## 11. Как воспроизвести (для ревьюера)

```bash
git checkout Blockwise-OAT
pip install -e .   # или uv sync

# Unit-тесты
pytest tests/test_blockwise_oat.py -v

# Синтетическая верификация обучения (без GPU/checkpoint)
python scripts/verify_blockwise_training.py --overfit-steps 120 --refine-iters 2

# С реальным checkpoint (пример)
python scripts/verify_blockwise_dataset.py \
  --policy-checkpoint path/to/ep-0950_sr-0.527.ckpt \
  --prefix-len 4 --prefix-samples 300

python scripts/verify_blockwise_policy_integration.py \
  --policy-checkpoint path/to/ep-0950_sr-0.527.ckpt \
  --tail-checkpoint path/to/blockwise_tail_decoder.pt \
  --prefix-len 4
```

---

## 12. Коммиты в ветке

| Коммит | Содержание |
|--------|------------|
| `175005e` | Первичная реализация Blockwise-OAT (AR prefix + ParallelTailDecoder) |
| *(текущий)* | Исправления по аудиту, verification-kit, расширенные тесты, review-документ |

---

*Документ подготовлен для ревью научного руководителя. Вопросы и замечания — через issues/PR в репозитории `GadzhiAskhabaliev/OAT-BLT-Dense`, ветка `Blockwise-OAT`.*
