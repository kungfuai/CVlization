#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
echo "Building docTR container..."
docker build -t doctr "$SCRIPT_DIR"

echo "Build complete! Image: doctr"
echo ""
echo "Usage:"
echo "  ./predict.sh [options]"
echo ""
echo "Example:"
echo "  ./predict.sh --image document.jpg"
echo "  ./predict.sh --image scan.jpg --format json"
