#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
echo "Building DeepSeek-OCR Docker image..."
docker build -t deepseek-ocr "$SCRIPT_DIR"

echo "Build complete! Image: deepseek-ocr"
echo ""
echo "Usage:"
echo "  ./predict.sh [options]"
echo ""
echo "Examples:"
echo "  ./predict.sh --image document.jpg"
echo "  ./predict.sh --image scan.jpg --task free_ocr"
echo "  ./predict.sh --image form.png --task grounding"
