#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building TurboDiffusion Docker image..."
echo "Note: This build compiles custom CUDA kernels and may take 10-20 minutes."
echo ""

docker build -t turbodiffusion "${SCRIPT_DIR}"

echo ""
echo "Build complete: turbodiffusion"
