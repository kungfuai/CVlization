#!/bin/bash
set -e

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_image> [options]"
    echo ""
    echo "Options:"
    echo "  --output <file>      Save output to file (optional)"
    echo "  --format <format>    Output format: markdown (default) or json"
    echo "  --device <device>    Device: cpu (default) or cuda"
    echo ""
    echo "Examples:"
    echo "  $0 document.png"
    echo "  $0 scan.jpg --format json --output result.json"
    echo "  $0 form.png --format markdown --device cuda"
    exit 1
fi

INPUT_FILE="$1"
shift  # Remove first argument, keep the rest as options

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get absolute path and filename
INPUT_DIR=$(cd "$(dirname "$INPUT_FILE")" && pwd)
INPUT_NAME=$(basename "$INPUT_FILE")

# Create outputs directory
mkdir -p "$(pwd)/outputs"

echo "Processing: $INPUT_FILE"
echo "Using Granite-Docling-258M VLM model..."
echo ""

# Run Docker container with GPU support
docker run --rm \
    --gpus all \
    -v "$INPUT_DIR:/app/inputs:ro" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    granite-docling \
    python predict.py "/app/inputs/$INPUT_NAME" "$@"

echo ""
echo "Done!"
