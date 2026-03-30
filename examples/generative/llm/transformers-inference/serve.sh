#!/usr/bin/env bash
# Launch an OpenAI-compatible server using HuggingFace Text Generation Inference (TGI).
# Pulls the official TGI image — no build step needed.
set -euo pipefail

MODEL_ID="${MODEL_ID:-allenai/Olmo-Hybrid-Instruct-DPO-7B}"
PORT="${PORT:-8080}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TGI_IMAGE="${TGI_IMAGE:-ghcr.io/huggingface/text-generation-inference:latest}"

echo "Starting TGI server (${MODEL_ID}) on port ${PORT}"
echo "OpenAI-compatible endpoint: http://localhost:${PORT}/v1"

docker run --rm --gpus all --ipc=host --shm-size 16g \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
  -p "${PORT}:80" \
  -v "${HF_CACHE}:/data" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${TGI_IMAGE}" \
  --model-id "${MODEL_ID}" \
  --trust-remote-code \
  --max-new-tokens 1024 \
  ${TGI_EXTRA_ARGS:-}
