#!/bin/bash

IMAGE_NAME="trl_sft"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .
echo "✅ Build complete! Image: $IMAGE_NAME"
