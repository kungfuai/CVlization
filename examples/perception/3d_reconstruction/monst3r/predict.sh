#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-monst3r}"

# Default arguments
# Use demo data from inside container (cloned during build)
INPUT_DIR="/opt/monst3r/demo_data/lady-running"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Parse command line arguments
EXTRA_ARGS=()
CUSTOM_INPUT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            CUSTOM_INPUT=true
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

# Determine input path for container
if [[ "$CUSTOM_INPUT" == "false" ]]; then
    # Use internal demo data (no mount needed)
    CONTAINER_INPUT_PATH="$INPUT_DIR"
    INPUT_MOUNT_ARGS=()
    echo "Running MonST3R inference..."
    echo "  Input: $INPUT_DIR (demo data inside container)"
else
    # Make custom input path absolute
    if [[ ! "$INPUT_DIR" = /* ]]; then
        INPUT_DIR="${SCRIPT_DIR}/${INPUT_DIR}"
    fi
    CONTAINER_INPUT_PATH="/workspace/data/input"
    INPUT_MOUNT_ARGS=(--mount "type=bind,src=${INPUT_DIR},dst=${CONTAINER_INPUT_PATH},readonly")
    echo "Running MonST3R inference..."
    echo "  Input: $INPUT_DIR (mounted from host)"
fi

# Make output path absolute
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

echo "  Output: $OUTPUT_DIR"
echo "  Image: $IMG"
echo ""

docker run --rm --gpus=all --shm-size 16G \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    "${INPUT_MOUNT_ARGS[@]}" \
    --mount "type=bind,src=${OUTPUT_DIR},dst=/workspace/outputs" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --mount "type=bind,src=${HOME}/.cache/torch,dst=/root/.cache/torch" \
    --env "PYTHONPATH=/cvlization_repo:/opt/monst3r" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMG" \
    python predict.py --input "$CONTAINER_INPUT_PATH" --output /workspace/outputs "${EXTRA_ARGS[@]}"
