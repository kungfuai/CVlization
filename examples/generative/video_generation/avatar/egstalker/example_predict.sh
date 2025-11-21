#!/usr/bin/env bash
# Example prediction script for EGSTalker
# This is a template - modify paths as needed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example usage - modify these paths
AUDIO_PATH="${AUDIO_PATH:-/workspace/input/audio.wav}"
REFERENCE_PATH="${REFERENCE_PATH:-/workspace/input/reference_dataset}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/egstalker_model}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"

echo "EGSTalker Example Inference"
echo "==========================="
echo "Audio: $AUDIO_PATH"
echo "Reference: $REFERENCE_PATH"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

bash "$SCRIPT_DIR/predict.sh" \
    --audio_path "$AUDIO_PATH" \
    --reference_path "$REFERENCE_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --custom_audio
