#!/bin/bash
set -e

docker run --rm --gpus all \
  -v /home/ubuntu/zz/CVlization/examples/doc_ai/moondream2_finetune:/workspace \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  moondream2-finetune \
  python finetune_moondream.py \
    --data data/captcha/train_subset_1000.jsonl \
    --val_data data/captcha/val.jsonl \
    --epochs 1 \
    --grad_accum 16 \
    --eval_steps 10 \
    --eval_samples 10
