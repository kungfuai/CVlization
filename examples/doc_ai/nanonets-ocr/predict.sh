#!/bin/bash
set -e

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_image> [options]"
    echo ""
    echo "Options:"
    echo "  --output <file>         Save output to file (optional)"
    echo "  --format <format>       Output format: markdown (default) or json"
    echo "  --mode <mode>           Processing mode: ocr (default) or vqa"
    echo "  --question <question>   Question for VQA mode (required with --mode vqa)"
    echo "  --device <device>       Device: cuda (default) or cpu"
    echo "  --max-tokens <num>      Maximum tokens to generate (default: 4096)"
    echo ""
    echo "Examples:"
    echo "  $0 document.png"
    echo "  $0 form.jpg --format json --output result.json"
    echo "  $0 chart.png --mode vqa --question 'What is the total revenue?'"
    echo "  $0 table.jpg --output output.md --device cuda"
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
echo "Using Nanonets-OCR2-3B VLM model..."
echo ""

# Run Docker container with GPU support
docker run --rm \
    --gpus all \
    -v "$INPUT_DIR:/app/inputs:ro" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    nanonets-ocr \
    python predict.py "/app/inputs/$INPUT_NAME" "$@"

echo ""
echo "Done!"
