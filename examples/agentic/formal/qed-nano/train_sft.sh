#!/usr/bin/env bash
# SFT fine-tuning for QED-Nano on FineProofs-SFT.
#
# Usage:
#   bash train_sft.sh                          # smoke test (30 steps, 100 samples)
#   bash train_sft.sh --max-steps 2000         # full training run
#   bash train_sft.sh --max-steps 2000 \
#     --push-to-hub your-org/qed-nano-sft      # upload checkpoint to HF Hub
#
# Environment variables:
#   MODEL_ID          Base model (default: Qwen/Qwen3-4B-Instruct-2507)
#   HF_TOKEN          HuggingFace token (optional for public models)
#   WANDB_API_KEY     Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: qed-nano-sft)
#   QED_SFT_IMAGE     Docker image name (default: qed-nano-sft)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
IMAGE="${QED_SFT_IMAGE:-qed-nano-sft}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image '${IMAGE}' not found. Building..."
  docker build -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile.sft" "${SCRIPT_DIR}"
fi

mkdir -p "$SCRIPT_DIR/outputs"

echo "Starting SFT training (image: ${IMAGE})"
docker run --rm --gpus all --ipc=host --shm-size 8g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-qed-nano-sft}" \
  "${IMAGE}" \
  python train_sft.py \
    --max-samples 100 \
    --max-steps 30 \
    "$@"
