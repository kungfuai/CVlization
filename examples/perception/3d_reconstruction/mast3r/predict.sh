#!/bin/bash
set -e

# Inference script for MASt3R
# Runs image matching and 3D reconstruction

IMAGE_NAME="mast3r"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Default to demo data inside container (NLE_tower from MASt3R repo)
INPUT_PATH="${SCRIPT_DIR}/data/images"
OUTPUT_PATH="${SCRIPT_DIR}/outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Check if input is default path and doesn't exist - use container demo data
USE_DEMO_DATA=false
if [ "$INPUT_PATH" = "${SCRIPT_DIR}/data/images" ] && [ ! -d "$INPUT_PATH" ]; then
    echo "No local input provided, using demo data from container..."
    USE_DEMO_DATA=true
    CONTAINER_INPUT="/opt/mast3r/assets/NLE_tower"
else
    CONTAINER_INPUT="/workspace/input"
fi

CONTAINER_OUTPUT="/workspace/output"

# Prepare docker run command
DOCKER_RUN="docker run --rm --gpus all -v ${WORK_DIR}:/mnt/cvl/workspace -e CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace} -e CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}"

# Mount input directory (unless using demo data)
if [ "$USE_DEMO_DATA" = false ]; then
    if [ ! -d "$INPUT_PATH" ]; then
        echo "Error: Input directory does not exist: $INPUT_PATH"
        exit 1
    fi
    DOCKER_RUN="${DOCKER_RUN} -v ${INPUT_PATH}:${CONTAINER_INPUT}:ro"
fi

# Create and mount output directory
mkdir -p "$OUTPUT_PATH"
DOCKER_RUN="${DOCKER_RUN} -v ${OUTPUT_PATH}:${CONTAINER_OUTPUT}"

echo "Running MASt3R inference..."
if [ "$USE_DEMO_DATA" = true ]; then
    echo "  Input: Demo data (NLE_tower, 7 images)"
else
    echo "  Input: $INPUT_PATH"
fi
echo "  Output: $OUTPUT_PATH"
echo ""

# Run inference
${DOCKER_RUN} ${IMAGE_NAME} \
    python3 /workspace/predict.py \
    --input ${CONTAINER_INPUT} \
    --output ${CONTAINER_OUTPUT} \
    ${EXTRA_ARGS}

echo ""
echo "✅ Inference complete!"
echo "📁 Results saved to: $OUTPUT_PATH"
