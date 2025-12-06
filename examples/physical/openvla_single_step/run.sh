#!/usr/bin/env bash
# Run OpenVLA single-step inference
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-openvla-inference:latest}"

# Default to interactive mode if no arguments
if [ $# -eq 0 ]; then
    echo "Running OpenVLA inference in interactive mode..."
    echo "Use --help for options"
    echo ""
    docker run --rm -it --gpus all \
        -v "${SCRIPT_DIR}/sample_images:/workspace/sample_images" \
        -v "${SCRIPT_DIR}/outputs:/workspace/outputs" \
        -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" \
        -e HF_HOME=/workspace/.cache/huggingface \
        "${IMAGE_NAME}" \
        python inference.py --interactive
else
    # Pass arguments through
    docker run --rm -it --gpus all \
        -v "${SCRIPT_DIR}/sample_images:/workspace/sample_images" \
        -v "${SCRIPT_DIR}/outputs:/workspace/outputs" \
        -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" \
        -e HF_HOME=/workspace/.cache/huggingface \
        "${IMAGE_NAME}" \
        python inference.py "$@"
fi
