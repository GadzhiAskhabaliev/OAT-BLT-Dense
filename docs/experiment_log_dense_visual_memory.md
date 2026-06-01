# Experiment log: Dense Visual Memory for OAT (LIBERO)

Живой журнал эксперимента для научрука / статьи (ablation section).  
Репозиторий: `BLT-OAT` (fork OAT). Бенчмарк: **LIBERO-10**, `N500` demos.

**Последнее обновление:** 2026-05-30 (UTC+3)

---

## 1. Гипотеза и смысл изменения

**Исходный OAT** подаёт в cross-attention короткую memory: pooled RGB-векторы (+ state в fused obs), длина порядка `n_obs_steps` (обычно **2 токена**). Пространственная структура кадра сжата в один вектор на timestep.

**Наша ветка (`use_dense_visual_memory=true`)** подаёт в тот же cross-attention decoder **длинную spatial memory**:
- patch-токены с каждой RGB-камеры на каждый obs-step;
- `time_embed` / `camera_embed`;
- опционально state-токены через MLP `state_to_memory`.

Мы **буквально льём в трансформер больше conditioning-токенов** (сотни вместо единиц). Это не «больше пикселей на входе», а **меньше сжатое** представление для attention.

**Ожидание:** модель может точнее локализовать объекты/гриппер → потенциально выше `success_rate`.  
**Риск:** оптимизация сложнее; в начале обучению «может поплохеть» (loss выше, медленнее сходимость, SR ниже), пока cross-attn и encoders не научатся отбирать релевантные patch’и.

---

## 2. Что изменили в архитектуре (vs default OAT)

| Компонент | Default OAT | Dense OAT (наш) |
|-----------|-------------|-----------------|
| RGB encoder | `RobomimicRgbEncoder` + global pool | `DenseRgbEncoder`: ResNet18 до `layer3`, **без** SpatialSoftmax/pool |
| Проекция в `d_model` | внутри pooled pipeline | `1×1 Conv` + `LayerNorm` на feature map |
| Memory length | `To` (≈2) | `2 камеры × To × L_patch + state_tokens` (L≈64 при 128×128) |
| State | в fused obs vector | отдельные memory-токены (`state_to_memory`) |
| Позиции memory | `cond_pos_emb` + `cond_emb` | dense: `memory_pos_emb` (`max_memory_len=1024`), `memory_is_embedded=True` |
| Cross-attn blocks | есть | **не меняли** |
| Action tokenizer (`OATTok`) | frozen | **не меняли** |
| Loss | CE по action tokens | **тот же** |

Ключевые файлы:
- `oat/perception/robomimic_vision_encoder.py` — `DenseRgbEncoder`
- `oat/policy/oatpolicy.py` — `get_dense_memory`, флаги абляций
- `oat/model/autoregressive/transformer_cache.py` — `_encode_memory`, `memory_pos_emb`
- `oat/config/train_oatpolicy.yaml` — hydra-флаги

Флаги абляций:
- `policy.use_dense_visual_memory` (alias: `use_cross_attn`)
- `policy.use_state_memory_tokens`
- `policy.use_task_uid_in_state_tokens`

---

## 3. Почему больший loss / MSE — не всегда «хуже»

Замечание научрука (A. Malinin): для dense-ветки **абсолютный train/val loss может быть выше**, чем у pooled OAT — это не обязательно регрессия.

Причины:
1. Другая размерность и статистика входа в decoder (длинная memory, другие pos emb).
2. Более богатое представление ≠ тот же CE-ландшафт, что у 2-токенного baseline.
3. **Целевая метрика проекта — `mean_success_rate` (rollout), не `reconst_mse` / `val_loss`.**  
   MSE/loss — sanity-check («учится / не взрывается»), финальное решение — SR на LIBERO eval.

---

## 4. Протокол валидации (короткий)

1. **Smoke / arch:** forward, backward, KV-cache dense vs legacy, shapes.
2. **Short A/B (N32, 1k steps):** legacy vs dense-варианты, одинаковый lr/batch.
3. **Long-run (N500):** полный датасет, `rollout_every=200`, best ckpt по `mean_success_rate`.

GO для long-run (short-run): dense **не хуже** legacy по val/reconst на стабильной конфигурации (допуск ~3–5%) **или** явно лучше по раннему mini-rollout.

---

## 5. Short A/B результаты (`libero10_N32_smoke`, 1k steps, 5 epochs)

| Конфиг | `val_loss` (final) | `test_reconst_mse` (final) | Комментарий |
|--------|-------------------:|---------------------------:|-------------|
| **legacy** (pooled) | 6.477 | 0.0132 | baseline |
| **dense + state + task_uid** | 7.531 | 0.1648 | сильная деградация |
| **dense_visual_only** (state off) | 6.848 | 0.0297 | лучше full dense+uid, хуже legacy |
| **dense + state, `task_uid` off** | **6.369** | **0.0133** | ≈ legacy по short metrics |

**Вывод short A/B:** проблема локализована в **`task_uid` в state memory**, не в dense visual tokens как таковых.

Run dirs (cluster): `output/smoke/ab_legacy_1k_r2`, `ab_dense_visual_only_1k`, `ab_dense_state_no_uid_1k`, `ab_dense_state_uid_1k_r2`.

---

## 6. Long-run (текущий)

**Активный run (dense + task_uid):** `oat_dense_with_uid_long_0530_220204`

| Поле | Значение |
|------|----------|
| Сессия tmux | `oat_dense_with_uid_long_0530_220204` |
| Train | stopped @ ~ep 804 for SR ladder eval |
| Checkpoints local | `ep-0300`, `ep-0500`, `ep-0700`, `latest.ckpt` |

| Поле (старый run без uid) | Значение |
|------|----------|
| Сессия tmux | `oat_dense_no_uid_long_0530_213112` |
| Конфиг | `dense + state`, `use_task_uid_in_state_tokens=false` |
| Датасет | `libero10_N500.zarr` (`training.num_demo=500`) |
| WandB | `logging.mode=disabled` (первый запуск упал: `No API key configured`) |
| Лог | `output/long/oat_dense_no_uid_long_0530_213112.log` |
| Hydra run | `output/long/oat_dense_no_uid_long_0530_213112/` |

**Снимок метрик (2026-05-30, во время обучения):**
- `global_step` ≈ 3136, `epoch` ≈ 6
- `train_loss` ≈ 4.96 (снижается с ~7 на старте)
- `val_loss` (epoch 0): 6.595 — ранняя точка, не финал
- `mean_success_rate`: **ещё нет** (первый rollout на `epoch % 200 == 0`)

Предыдущая tmux-сессия `oat_dense_no_uid_long_0530_212603` — упала на init wandb.

---

## 7. Если `success_rate` сильно просядет — шаблон для ablation (статья / тикет)

Ниже — **готовые причины**, которые можно вставить в Discussion/Ablation, если dense < baseline OAT по SR после полного обучения.

### 7.1. Усложнение optimization (главная гипотеза)

- Cross-attention получает **~200–300+ memory tokens** вместо 2; effective capacity растёт, но gradient signal размазывается по patch’ам.
- Новые модули (`DenseRgbEncoder`, `memory_pos_emb`, time/camera embed) учатся с нуля при frozen tokenizer → дольше «сходимость» closed-loop политики.
- На ранних эпохах политика может вести себя хуже baseline при нормальном или даже падающем train loss.

### 7.2. Несопоставимость offline-метрик

- `reconst_mse` / `val_loss` слабо коррелируют с `success_rate` (особенно при смене conditioning).
- Возможен сценарий: loss ≈ baseline, SR ниже — или loss выше, SR сопоставим (как отмечал научрук).

### 7.3. `task_uid` в memory (подтверждено абляцией)

- Включение `task_uid` embedding в state tokens резко ухудшило short-run (`mse` 0.013 → 0.165).
- Возможные механизмы: неверный id range/семантика, конфликт с state MLP, переобучение на spurious task signal.
- Long-run идёт **без** `task_uid`; если SR всё равно низкий — причина не только в uid.

### 7.4. Inductive bias pooled OAT

- Global pooling даёт компактный, устойчивый вектор — для некоторых LIBERO задач этого может хватать.
- Dense memory полезен, когда нужна пространственная привязка; если baseline уже «выдавливает» SR из pool, выигрыш может быть < overhead.

### 7.5. Compute / eval budget

- Меньше эффективных «эпох» на единицу wall-clock (тяжелее forward).
- Редкий rollout (`rollout_every=200`) → поздняя обратная связь по SR; best checkpoint по SR может не совпасть с best по loss.

### 7.6. Гиперпараметры не перенастроены

- Те же `policy_lr`, `obs_enc_lr`, batch=256, что у pooled baseline — для длинной memory может быть suboptimal (нужен отдельный sweep).

**Формулировка для тикета (кратко):**  
> Dense memory увеличивает информационную ёмкость conditioning, но усложняет обучение cross-attention. Наблюдаемый/ожидаемый провал SR не обязательно означает ошибку реализации: он может следовать из optimization difficulty, слабой корреляции offline-метрик с SR, и подтверждённой деградации от `task_uid` (исключена в production run). Финальная оценка — только paired rollout на LIBERO-10 с теми же seeds и eval protocol, что baseline.

---

## 8. Git / деплой

| Commit | Описание |
|--------|----------|
| (ранние) | DenseRgbEncoder, OATPolicy memory, decoder `memory_pos_emb` |
| `9c305cd` | fix `lr_scheduler` imports (diffusers) |
| `020f63d` | `use_state_memory_tokens` |
| `9481ae0` | `use_task_uid_in_state_tokens` |

Синк на кластер: `rsync` в `~/OAT-RoboMimic-Fine-tune/BLT-OAT/`, **exclude `data/`**.

Docker: `oat_robomimic_askhabaliev_gs`.  
Tokenizer ckpt: `tokenizer_ep-0950_mse-0.002.ckpt` (HF Mirageinv/oat) — **без изменений**.

---

## 9. Как обновлять этот лог и анализировать все run'ы

Каждый Hydra-run пишет в свою папку:
- `logs.json` — train_loss, val_loss, test_reconst_mse, mean_success_rate (на rollout-эпохах)
- `.hydra/overrides.yaml` — конфиг (embed_dim, task_uid, …)
- `checkpoints/` — ckpt для eval и attention viz
- `*.log` — tee-лог в `output/long/`

**Манифест всех run'ов:** `output/long/RUN_MANIFEST.md` (append при sweep)

**Сводка по всем run'ам:**
```bash
python scripts/summarize_training_runs.py --root output/long
# -> output/long/summary/all_runs_summary.csv + .md
```

**Графики loss / MSE / SR:**
```bash
python scripts/plot_training_runs.py \
  output/long/oat_dense_with_uid_long_0530_220204 \
  output/long/dense_emb128_with_uid_* \
  --out output/long/summary/plots
```

**Rollout eval (success rate) из checkpoint:**
```bash
python scripts/eval_policy_sim.py \
  -c output/long/<run>/checkpoints/latest.ckpt \
  -o eval/<run_name>

# Ladder 300/500/700 (paired SR): docs/plans/libero_sr_eval_ladder_300_500_700.md
# PHASE=A bash scripts/cluster/run_ladder_sr_eval.sh
```

**Cross-attention heatmaps:**
```python
from oat.common.attention_viz import visualize_cross_attention
# policy from checkpoint + sample batch -> overlay on RGB
```

Команды мониторинга (cluster):

```bash
tmux ls
tmux attach -t oat_dense_no_uid_long_0530_213112
tail -f ~/OAT-RoboMimic-Fine-tune/BLT-OAT/output/long/oat_dense_no_uid_long_0530_213112.log
python3 -c "
import json; from pathlib import Path
p=Path('~/OAT-RoboMimic-Fine-tune/BLT-OAT/output/long/oat_dense_no_uid_long_0530_213112/logs.json').expanduser()
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for k in ['mean_success_rate','val_loss','train_loss']:
    r=[x for x in rows if k in x]
    if r: print(k, r[-1])
"
```

---

## 10. Changelog

### 2026-05-31 — HF checkpoint upload watcher (epochs 300 & 500, train continues)
- Script: `scripts/watch_hf_checkpoint_upload.py`
- Launcher: `scripts/cluster/launch_hf_upload_tmux.sh`
- Default repo: `hackhackhack66666/OAT-BLT-LIBERO-300` (hub API; git-xet fallback)
- Checkpoint ladder HF repos:
  - 300 → `hackhackhack66666/OAT-BLT-LIBERO-300` ✅
  - 500 → `hackhackhack66666/OAT-BLT-LIBERO-500` ✅
  - 700 → `hackhackhack66666/OAT-BLT-Libero-700` ✅
  - 900, 1100 → TBD (`OAT-BLT-Libero-900`, `OAT-BLT-Libero-1100`)
- Each HF repo: `README.md` (EN) + `training_metrics_dashboard.png` + this experiment log
- SR eval plan: `docs/plans/libero_sr_eval_ladder_300_500_700.md` (sim eval; ckpts kept on disk until done)
- `--target-epochs` + `--epoch-repo EPOCH=REPO` per snapshot; train continues
- Trigger when `epoch > target` (so ep-N ckpt exists); train not stopped
- Requires `HF_TOKEN`; logs: `hf_upload_report.jsonl`, `hf_upload_watch.log`

### 2026-05-31 — Counterfactual early-stop watcher (no real stop)
- Добавлен CPU-only watcher: `scripts/watch_early_stop_report.py`.
- Добавлен launcher: `scripts/cluster/launch_early_stop_watch_tmux.sh`.
- Watcher проверяет с `epoch >= 0` (ранние ep помечаются как noisy в отчёте), каждый час, пишет:
  - `early_stop_report.jsonl` (история проверок),
  - `early_stop_report.md` (сводка для статьи).
- Важно: watcher **не останавливает** обучение, только пишет где **был бы** стоп.

### 2026-05-30 — Sweep relaunch + post-hoc tooling
- Скрипты: `summarize_training_runs.py`, `plot_training_runs.py`, `RUN_MANIFEST.md`.
- Sweep на GPU1 в tmux, `USE_TASK_UID=true` (как основной run).

### 2026-05-30 — Перезапуск с `task_uid`
- Остановлен long-run без uid (~epoch 15) и embed-dim sweep.
- Новый long-run: `dense + state + task_uid=true`, N500, tmux `oat_dense_with_uid_long_*`.

### 2026-05-30 — Overnight embed-dim sweep (GPU1)
- Запущен последовательный sweep `embed_dim` ∈ {128, 384, 512} на свободной GPU1.
- Baseline 256 остаётся на GPU0 (`oat_dense_no_uid_long_0530_213112`).
- Скрипт: `scripts/cluster/overnight_dense_embed_dim_sweep.sh`, tmux: `oat_dense_embed_sweep_night`.

### 2026-05-30 — Short A/B завершён, long-run запущен
- Реализованы dense memory + абляционные флаги.
- Short A/B: стабильная конфигурация **`dense + state, без task_uid`**.
- Long-run N500 в tmux `oat_dense_no_uid_long_0530_213112`, wandb off.
- Train идёт: к ~epoch 6 `train_loss` ~4.96; SR rollout ещё не было.
- Создан этот experiment log для статьи/тикета.

### 2026-05-30 — Первый long-run упал
- tmux `oat_dense_no_uid_long_0530_212603`: `wandb.errors.UsageError: No API key`.
- Перезапуск с `logging.mode=disabled`.

---

## 11. Overnight sweep: `embed_dim` / `dense_feature_dim`

**Мотивация:** при фиксированной spatial resolution (L patch’ей) меняется только ширина token embedding — влияет на ёмкость visual memory и нагрузку на cross-attn.

**Ограничение кода:** `dense_feature_dim` должен совпадать с `policy.embed_dim` (иначе `dense memory expects cond dim {n_emb}`).

| GPU | Задача | `embed_dim` | batch |
|-----|--------|------------:|------:|
| 0 | основной long-run | 256 | 256 |
| 1 | sweep (последовательно) | 128 → 384 → 512 | 256 / 192 / 128 |

Скрипт: `scripts/cluster/overnight_dense_embed_dim_sweep.sh`  
tmux (cluster): `oat_dense_embed_sweep_night`  
Конфиг как у основного run: dense + state, **без** `task_uid`, `logging.mode=disabled`.

---

## 12. Открытые вопросы

1. Финальный **SR dense vs legacy** на N500 после `rollout_every`.
2. Какой `embed_dim` лучше по SR / sample efficiency (sweep 128/256/384/512)?
3. Нужен ли отдельный sweep lr / warmup для dense encoders?
4. Как корректно вернуть `task_uid` (mapping id, gating, не в state MLP)?
5. Attention viz на лучшем ckpt — куда смотрит policy на успехе/провале?

---

## 13. Ссылки

- План реализации: [`docs/plans/cross-attn_dense_oat.md`](plans/cross-attn_dense_oat.md)
- Attention viz: `oat/common/attention_viz.py`
