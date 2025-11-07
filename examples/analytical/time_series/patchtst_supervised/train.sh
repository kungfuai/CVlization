#!/bin/bash
set -e

# Run training in Docker container
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v /home/ubuntu/.cache:/root/.cache \
  patchtst_supervised \
  python train.py "$@"
