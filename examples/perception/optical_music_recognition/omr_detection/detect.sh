#!/usr/bin/env bash
# Run detection + cell derivation on one page image, inside the
# omr-detection Docker image.
#
#   detect.sh --image /tmp/det_l7a_500/images/L7a_04000_p1.png \
#             --checkpoint outputs/detector_l7a_500_wide/run/weights/best.pt \
#             --crops-dir /tmp/crops --inspect 12
#
# The image's parent dir is mounted at /img; --checkpoint and --crops-dir
# resolve inside /workspace unless absolute under the mounted dirs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/omr-detection:latest}"

IMAGE=""
CROPS_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --crops-dir) CROPS_DIR="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "Usage: detect.sh --image <png> --checkpoint <path> [...]" >&2
  exit 1
fi

IMG_DIR_ABS="$(cd "$(dirname "$IMAGE")" && pwd)"
IMG_NAME="$(basename "$IMAGE")"
ARGS+=("--image" "/img/${IMG_NAME}")

CROPS_MOUNT=()
if [[ -n "$CROPS_DIR" ]]; then
  mkdir -p "$CROPS_DIR"
  CROPS_DIR_ABS="$(cd "$CROPS_DIR" && pwd)"
  CROPS_MOUNT=(--mount "type=bind,src=${CROPS_DIR_ABS},dst=/crops")
  ARGS+=("--crops-dir" "/crops")
fi

docker run --rm --gpus all \
  --workdir /workspace \
  --shm-size=8g \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${IMG_DIR_ABS},dst=/img,readonly" \
  "${CROPS_MOUNT[@]}" \
  --env "PYTHONPATH=/cvlization_repo:/workspace" \
  --env "PYTHONUNBUFFERED=1" \
  --env "YOLO_VERBOSE=False" \
  "$IMG" python3 pipeline.py "${ARGS[@]}"
