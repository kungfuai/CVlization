#!/usr/bin/env bash
# SFT off safckylj at the per-measure level inside the per-measure image.
#
#   train.sh --data /tmp/per_measure_v1 --vlm-ckpt /vlm_sft/outputs/safckylj/final_model \
#            --output outputs/per_measure_sft_v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLM_SFT_DIR="$(cd "$SCRIPT_DIR/../vlm_omr_sft" && pwd)"
DET_DIR="$(cd "$SCRIPT_DIR/../omr_detection" && pwd)"
IMG="${CVL_IMAGE:-cvlization/vlm-omr-per-measure-sft:latest}"

DATA_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; ARGS+=("--data" "/data"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: train.sh --data <dir> [--vlm-ckpt ...] [...]" >&2; exit 1
fi
DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"

mkdir -p /tmp/hf_user2

docker run --rm --gpus all --shm-size=8g \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${VLM_SFT_DIR},dst=/vlm_sft" \
  --mount "type=bind,src=${DET_DIR},dst=/det" \
  --mount "type=bind,src=${DATA_DIR_ABS},dst=/data,readonly" \
  --mount "type=bind,src=/tmp/hf_user2,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo:/workspace:/vlm_sft:/det" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=/root/.cache/huggingface" \
  "$IMG" python3 train_per_measure.py "${ARGS[@]}"
