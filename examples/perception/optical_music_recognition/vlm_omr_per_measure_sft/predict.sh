#!/usr/bin/env bash
# End-to-end per-measure OMR prediction inside the per-measure Docker
# image. Detector + per-measure VLM both run from the same container.
#
#   predict.sh --image page.png \
#              --det-ckpt /det/outputs/detector_v3_best.pt \
#              --vlm-ckpt outputs/per_measure_fresh_v4/final_model \
#              --out page.mxc2
#
# The image's parent dir is mounted at /img. --det-ckpt and --vlm-ckpt
# resolve inside /det and /workspace respectively (or absolute under
# the mounted dirs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLM_SFT_DIR="$(cd "$SCRIPT_DIR/../vlm_omr_sft" && pwd)"
DET_DIR="$(cd "$SCRIPT_DIR/../omr_detection" && pwd)"
IMG="${CVL_IMAGE:-cvlization/vlm-omr-per-measure-sft:latest}"

IMAGE=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "$IMAGE" ]]; then
  echo "Usage: predict.sh --image <png> --det-ckpt <pt> --vlm-ckpt <dir> [--out <path>] [--verbose]" >&2
  exit 1
fi
IMG_DIR_ABS="$(cd "$(dirname "$IMAGE")" && pwd)"
IMG_NAME="$(basename "$IMAGE")"
ARGS+=("--image" "/img/${IMG_NAME}")

mkdir -p /tmp/hf_user2

docker run --rm --gpus all --shm-size=8g \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${VLM_SFT_DIR},dst=/vlm_sft" \
  --mount "type=bind,src=${DET_DIR},dst=/det" \
  --mount "type=bind,src=${IMG_DIR_ABS},dst=/img,readonly" \
  --mount "type=bind,src=/tmp/hf_user2,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo:/workspace:/vlm_sft:/det" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  "$IMG" python3 pipeline_per_measure.py "${ARGS[@]}"
