#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-map_anything}"

# Default arguments
IMAGES_DIR="${SCRIPT_DIR}/data/images"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --images)
            IMAGES_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            # Pass through any other arguments to predict.py
            break
            ;;
    esac
done

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"

# If IMAGES_DIR is relative, make it absolute
if [[ ! "$IMAGES_DIR" = /* ]]; then
    IMAGES_DIR="${SCRIPT_DIR}/${IMAGES_DIR}"
fi

# If OUTPUT_DIR is relative, make it absolute
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

echo "Running MapAnything inference..."
echo "  Images: $IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Image: $IMG"
echo ""

docker run --rm --gpus=all --shm-size 16G \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${IMAGES_DIR},dst=/workspace/data/images,readonly" \
    --mount "type=bind,src=${OUTPUT_DIR},dst=/workspace/output" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMG" \
    python predict.py --images /workspace/data/images --output /workspace/output "$@"
