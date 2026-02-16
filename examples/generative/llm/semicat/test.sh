#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
IMG="${CVL_IMAGE:-semicat}"

mkdir -p "$HOME/.cache/cvlization"

# Smoke test: 50 steps on Text8 with small data subset
docker run --rm --gpus=all --shm-size 16G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/cache" \
    --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
    --env "PYTHONPATH=/opt/semicat" \
    --env "PROJECT_ROOT=/opt/semicat" \
    --env "CVL_CACHE_DIR=/cache" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMG" \
    python /workspace/train.py --steps 50 --batch-size 32 \
        data.small_run=true \
        +trainer.max_steps=50 \
        trainer.val_check_interval=50 \
        +model.sd_prop=0.0 \
        test=false \
        "$@"
