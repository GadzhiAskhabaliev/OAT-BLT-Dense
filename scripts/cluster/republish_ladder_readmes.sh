#!/usr/bin/env bash
# Update README.md on all ladder HF repos: training + sim eval dashboards inline.
set -euo pipefail

if [[ -z "${REPO_HOST:-}" ]]; then
  if [[ -d /workspace/OAT-RoboMimic-Fine-tune/BLT-OAT ]]; then
    REPO_HOST="/workspace/OAT-RoboMimic-Fine-tune/BLT-OAT"
  else
    REPO_HOST="${HOME}/OAT-RoboMimic-Fine-tune/BLT-OAT"
  fi
fi

RUN_DIR="${RUN_DIR:-output/long/oat_dense_with_uid_long_0530_220204}"
EVAL_ROOT="${EVAL_ROOT:-output/eval/ladder_screen_pt30}"
CONFIRM_ROOT="${CONFIRM_ROOT:-output/eval/ladder_confirm_pt50}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN" >&2
  exit 1
fi

cd "$REPO_HOST"
mkdir -p "$RUN_DIR/checkpoints"

# Restore upload_meta from HF (deleted locally after upload)
python3 << 'PY'
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ladder = [
    (300, "hackhackhack66666/OAT-BLT-LIBERO-300"),
    (500, "hackhackhack66666/OAT-BLT-LIBERO-500"),
    (700, "hackhackhack66666/OAT-BLT-Libero-700"),
]
ckpt_dir = Path("output/long/oat_dense_with_uid_long_0530_220204/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
for epoch, repo in ladder:
    name = f"ep-{epoch:04d}_upload_meta.json"
    dest = ckpt_dir / name
    if dest.is_file():
        continue
    src = hf_hub_download(repo, name, repo_type="model")
    shutil.copy(src, dest)
    print("meta", name)
PY

python scripts/publish_ladder_hf_readmes.py \
  --run-dir "$RUN_DIR" \
  --eval-root "$EVAL_ROOT" \
  --confirm-root "$CONFIRM_ROOT" \
  --out output/hf_ladder_publish \
  --readme-only

echo "README updated on all ladder repos"
