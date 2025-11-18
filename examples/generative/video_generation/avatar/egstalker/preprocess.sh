#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE="${CVL_IMAGE:-egstalker}"

docker run --rm \
  --gpus=all \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/host" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache,dst=/root/.cache" \
  -e PYTHONPATH=/workspace/host:/workspace/egstalker:/cvlization_repo \
  -e PYTHONUNBUFFERED=1 \
  --workdir /workspace/host \
  "${IMAGE}" \
  python data_utils/process.py "$@"
