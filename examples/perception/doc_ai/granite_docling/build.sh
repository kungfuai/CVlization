#!/bin/bash
set -e

# Build the Granite-Docling Docker image
echo "Building Granite-Docling container..."
docker build -t granite-docling .

echo "Build complete! Image: granite-docling"
echo ""
echo "Usage:"
echo "  ./predict.sh <input_image> [options]"
echo ""
echo "Example:"
echo "  ./predict.sh document.png"
echo "  ./predict.sh scan.jpg --format json"
