#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
echo "Building dots.ocr container..."
docker build -t dots_ocr "$SCRIPT_DIR"

echo "Build complete! Image: dots_ocr"
echo ""
echo "Usage:"
echo "  ./predict.sh [options]"
echo ""
echo "Example:"
echo "  ./predict.sh --image document.jpg"
echo "  ./predict.sh --image scan.jpg --format json"
