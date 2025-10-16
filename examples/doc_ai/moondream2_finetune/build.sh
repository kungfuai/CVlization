#!/bin/bash
set -e

IMAGE_NAME="moondream2-finetune"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo ""
echo "Build complete! Image: $IMAGE_NAME"
echo ""
echo "Usage:"
echo "  ./train.sh --data /path/to/data.jsonl"
echo ""
echo "Example:"
echo "  ./train.sh --data sample_data.jsonl --epochs 2 --batch-size 6"
