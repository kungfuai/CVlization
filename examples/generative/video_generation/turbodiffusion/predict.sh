#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Default values
PROMPT="A cat playing piano in a jazz club, cinematic lighting"
OUTPUT="outputs/generated_video.mp4"
NUM_STEPS=4
RESOLUTION="480p"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--prompt TEXT] [--output PATH] [--num_steps 1-4] [--resolution 480p|720p] [--seed N]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${SCRIPT_DIR}/$(dirname "${OUTPUT}")"

echo "Running TurboDiffusion inference..."
echo "Prompt: ${PROMPT}"
echo "Output: ${OUTPUT}"
echo "Steps: ${NUM_STEPS}"
echo "Resolution: ${RESOLUTION}"
echo ""

docker run --rm --gpus all \
    -v "${SCRIPT_DIR}/outputs:/workspace/outputs" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${WORK_DIR}:/mnt/cvl/workspace" \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    turbodiffusion \
    python /workspace/predict.py \
        --prompt "${PROMPT}" \
        --output "/workspace/${OUTPUT}" \
        --num_steps "${NUM_STEPS}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}"

echo ""
echo "Video saved to: ${SCRIPT_DIR}/${OUTPUT}"
