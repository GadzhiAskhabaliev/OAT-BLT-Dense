# Counterfactual Early-Stop Report

- Run dir: `/home/askhabaliev_gs/OAT-RoboMimic-Fine-tune/BLT-OAT/output/long/oat_dense_with_uid_long_0530_220204`
- Updated: `2026-05-31T10:28:48+00:00`
- Mode: `counterfactual_only` (no real stop, no kill signal)

## Checks

| Time (UTC) | Epoch | Verdict | Reason |
|---|---:|---|---|
| 2026-05-30T21:23:02+00:00 | 78 | waiting_min_epoch | epoch=78 < min_epoch=150 |
| 2026-05-30T21:28:39+00:00 | 81 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-30T21:28:39+00:00 | 81 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-30T22:28:39+00:00 | 116 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-30T23:28:40+00:00 | 151 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T00:28:40+00:00 | 187 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T01:28:41+00:00 | 222 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T02:28:41+00:00 | 257 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T03:28:42+00:00 | 293 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T04:28:43+00:00 | 328 | would_stop_overfit | counterfactual stop: overfit pattern on recent val points |
| 2026-05-31T05:28:43+00:00 | 363 | continue | no counterfactual stop signal |
| 2026-05-31T06:28:44+00:00 | 399 | continue | no counterfactual stop signal |
| 2026-05-31T07:28:45+00:00 | 434 | would_review_plateau | counterfactual review: offline metrics plateau |
| 2026-05-31T08:28:46+00:00 | 469 | would_review_plateau | counterfactual review: offline metrics plateau |
| 2026-05-31T09:28:47+00:00 | 505 | would_review_plateau | counterfactual review: offline metrics plateau |
| 2026-05-31T10:28:48+00:00 | 540 | would_review_plateau | counterfactual review: offline metrics plateau |

## Trigger Counts

- `continue`: 2
- `waiting_min_epoch`: 1
- `would_review_plateau`: 4
- `would_stop_overfit`: 9
