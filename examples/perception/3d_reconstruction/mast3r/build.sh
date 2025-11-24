#!/bin/bash
set -e

# Build script for MASt3R inference example

IMAGE_NAME="mast3r"

echo "Building MASt3R Docker image..."
docker build -t ${IMAGE_NAME} .

echo ""
echo "✅ Build complete!"
echo "Image: ${IMAGE_NAME}"
echo ""
echo "Next steps:"
echo "  ./predict.sh                    # Run inference on demo data"
echo "  ./predict.sh --input /path/to/images  # Run on custom images"
