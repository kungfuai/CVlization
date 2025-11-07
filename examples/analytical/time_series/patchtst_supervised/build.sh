#!/bin/bash
set -e

# Build Docker image
docker build -t patchtst_supervised .

echo "Build complete! Image: patchtst_supervised"
