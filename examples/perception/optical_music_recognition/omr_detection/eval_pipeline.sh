#!/usr/bin/env bash
# Run the end-to-end pipeline evaluation inside the combined
# (vlm-omr-sft + ultralytics) image.
#
#   eval_pipeline.sh --data /tmp/det_l7a_eval \
#       --det-ckpt outputs/detector_l7a_500_wide/run/weights/best.pt \
#       --vlm-ckpt /vlm_sft/outputs/safckylj/final_model \
#       --n 10 --mode page
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VLM_SFT_DIR="$(cd "$SCRIPT_DIR/../vlm_omr_sft" && pwd)"
IMG="${CVL_IMAGE:-cvlization/omr-pipeline:latest}"

DATA_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; ARGS+=("--data" "/data"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: eval_pipeline.sh --data <dir> --det-ckpt <path> --vlm-ckpt <path> [...]" >&2
  exit 1
fi
DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"
mkdir -p /tmp/hf_user

docker run --rm --gpus all \
  --workdir /workspace \
  --shm-size=8g \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${VLM_SFT_DIR},dst=/vlm_sft" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${DATA_DIR_ABS},dst=/data,readonly" \
  --mount "type=bind,src=/tmp/hf_user,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo:/workspace:/vlm_sft" \
  --env "PYTHONUNBUFFERED=1" \
  --env "YOLO_VERBOSE=False" \
  --env "HF_HOME=/root/.cache/huggingface" \
  "$IMG" python3 eval_pipeline.py "${ARGS[@]}"
