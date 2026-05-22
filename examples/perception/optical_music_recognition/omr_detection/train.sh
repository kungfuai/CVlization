#!/usr/bin/env bash
# Train the layout detector inside the omr-detection Docker image.
#
#   train.sh --data /tmp/det_l7a_50 --config configs/detector_l7a.yaml
#
# Mounts: this folder at /workspace; the data dir at /data; outputs go
# to ./outputs/ on the host (created if absent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/omr-detection:latest}"

# Parse --data out of $@ so we can mount it.
DATA_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; ARGS+=("--data" "/data"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: train.sh --data <dir> [--config configs/...]" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/outputs"
DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"

docker run --rm --gpus all \
  --workdir /workspace \
  --shm-size=8g \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${DATA_DIR_ABS},dst=/data,readonly" \
  --env "PYTHONPATH=/cvlization_repo:/workspace" \
  --env "PYTHONUNBUFFERED=1" \
  --env "YOLO_VERBOSE=False" \
  "$IMG" python3 train_detector.py "${ARGS[@]}"
