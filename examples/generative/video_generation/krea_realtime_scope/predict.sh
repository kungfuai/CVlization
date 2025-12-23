#!/bin/bash
set -e

# Run Krea Realtime video generation via Scope framework
IMAGE_NAME=${IMAGE_NAME:-"krea_realtime_scope"}
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"

# Configuration
PROMPT=${PROMPT:-"A serene mountain lake at sunset with reflections"}
WIDTH=${WIDTH:-832}
HEIGHT=${HEIGHT:-480}
NUM_BLOCKS=${NUM_BLOCKS:-9}
SEED=${SEED:-42}
FPS=${FPS:-24}
QUANTIZATION=${QUANTIZATION:-none}
OUTPUT_DIR="${EXAMPLE_DIR}/outputs"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Display configuration
echo "=================================================="
echo "Krea Realtime Video Generation (via Scope)"
echo "=================================================="
echo "Prompt: ${PROMPT}"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Blocks: ${NUM_BLOCKS} (~${NUM_BLOCKS} seconds at ${FPS}fps)"
echo "Seed: ${SEED}"
echo "Quantization: ${QUANTIZATION}"
echo "Output: ${OUTPUT_DIR}/video.mp4"
echo "=================================================="
echo ""

# Run inference
docker run --rm \
    --gpus=all \
    -v ~/.cache/cvlization:/root/.cache \
    -v "${OUTPUT_DIR}":/workspace/outputs \
    -v "${WORK_DIR}:/mnt/cvl/workspace" \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    "${IMAGE_NAME}" \
    python3 predict.py \
        --prompt "${PROMPT}" \
        --output outputs/video.mp4 \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --num-blocks "${NUM_BLOCKS}" \
        --seed "${SEED}" \
        --fps "${FPS}" \
        --quantization "${QUANTIZATION}"

echo ""
echo "✅ Video saved to: ${OUTPUT_DIR}/video.mp4"
