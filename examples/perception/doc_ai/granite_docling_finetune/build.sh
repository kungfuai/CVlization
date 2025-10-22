#!/bin/bash
set -e

echo "Building Granite-Docling fine-tuning container..."
docker build -t granite-docling-finetune .

echo ""
echo "Build complete! Image: granite-docling-finetune"
echo ""
echo "Usage:"
echo "  ./train.sh [options]"
echo ""
echo "Example:"
echo "  ./train.sh --train-data /path/to/data.jsonl --epochs 3"
