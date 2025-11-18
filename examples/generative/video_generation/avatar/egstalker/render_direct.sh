#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm --gpus=all \
  --workdir /workspace/host \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/host" \
  --env "PYTHONPATH=/workspace/host:/workspace/egstalker" \
  egstalker \
  python /workspace/egstalker/render.py \
    -s /workspace/host/data/test_videos \
    --model_path /workspace/host/output/test_bfm \
    --iteration 10000 \
    --batch 16 \
    --skip_train \
    --skip_test \
    --custom_aud aud_short.npy \
    --custom_wav aud_short.wav \
    "$@"
