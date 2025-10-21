#!/bin/bash
set -e

IMAGE_NAME="gpt_oss_grpo"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "✅ Build complete! Image: $IMAGE_NAME"
