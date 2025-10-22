#!/bin/bash
# Wrapper script to run prediction inside Docker container

docker run --shm-size 16G --runtime nvidia -it --rm \
    -v $(pwd):/workspace \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    wan2gp \
    python predict.py "$@"
