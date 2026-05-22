#!/usr/bin/env bash
# Evaluate the layout detector inside the omr-detection Docker image.
#
#   eval.sh --data /tmp/det_l7a_500 \
#           --checkpoint outputs/detector_l7a_500_wide/run/weights/best.pt
#
# --checkpoint is resolved inside /workspace (this folder).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/omr-detection:latest}"

DATA_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; ARGS+=("--data" "/data"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: eval.sh --data <dir> --checkpoint <path> [...]" >&2
  exit 1
fi
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
  "$IMG" python3 eval_detector.py "${ARGS[@]}"
