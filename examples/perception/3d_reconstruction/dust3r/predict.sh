#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-dust3r}"

# Default arguments - use hunyuanworld_mirror images to avoid duplication
INPUT_DIR="${SCRIPT_DIR}/../hunyuanworld_mirror/data/images"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            # Pass through any other arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Make paths absolute
if [[ ! "$INPUT_DIR" = /* ]]; then
    INPUT_DIR="${SCRIPT_DIR}/${INPUT_DIR}"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

echo "Running DUSt3R inference..."
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Image: $IMG"
echo ""

docker run --rm --gpus=all --shm-size 16G \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${INPUT_DIR},dst=/workspace/data/images,readonly" \
    --mount "type=bind,src=${OUTPUT_DIR},dst=/workspace/outputs" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --env "PYTHONPATH=/cvlization_repo:/opt/dust3r" \
    --env "PYTHONUNBUFFERED=1" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    "$IMG" \
    python predict.py --input /workspace/data/images --output /workspace/outputs "${EXTRA_ARGS[@]}"
