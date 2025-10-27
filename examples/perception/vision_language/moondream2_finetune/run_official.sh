#!/bin/bash
set -e

docker run --rm --gpus all \
  -v /home/ubuntu/zz/CVlization/examples/doc_ai/moondream2_finetune:/workspace \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  moondream2-finetune \
  python finetune_moondream.py "$@"
