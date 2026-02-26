#!/usr/bin/env bash
# RL (GRPO) training for QED-Nano using PipelineRL.
#
# PipelineRL runs all components in a single container on a single node:
#   - Actor vLLM servers   (ACTOR_GPUS GPUs)
#   - Preprocessor workers
#   - DeepSpeed trainer    (TRAINER_GPUS GPUs)
#   - Proof grader client  (calls external OpenAI-compatible API)
#
# Minimum requirements: 4 NVIDIA GPUs + an OpenAI-compatible grader endpoint.
#
# Usage:
#   # Build image first
#   bash train_rl.sh --build
#
#   # 4-GPU run with OpenAI grader
#   OPENAI_API_KEY=sk-... bash train_rl.sh
#
#   # 8-GPU run with local vLLM grader (run cvl-vllm serve.sh separately)
#   ACTOR_GPUS=4 TRAINER_GPUS=4 \
#   GRADER_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
#   OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
#   OPENAI_API_KEY=token \
#   bash train_rl.sh
#
#   # Start from an SFT checkpoint
#   MODEL_ID=outputs/sft \
#   OPENAI_API_KEY=sk-... \
#   bash train_rl.sh
#
# Environment variables:
#   MODEL_ID          Base model or SFT checkpoint path  (default: Qwen/Qwen3-4B-Instruct-2507)
#   MODEL_REVISION    HF revision                        (default: main)
#   ACTOR_GPUS        GPUs for vLLM rollout servers      (default: 2)
#   TRAINER_GPUS      GPUs for DeepSpeed trainer         (default: 2)
#   SEQ_LENGTH        Max context length                 (default: 8192)
#   GRADER_MODEL      Grader model name                  (default: gpt-4o-mini)
#   OPENAI_API_KEY    API key for grader (required)
#   OPENAI_BASE_URL   Grader API base URL                (default: OpenAI)
#   WANDB_API_KEY     Enable W&B logging
#   OUTPUT_DIR        Checkpoint directory               (default: outputs/rl_<timestamp>)
#   QED_TRAIN_IMAGE   Docker image name                  (default: qed-nano-train)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
IMAGE="${QED_TRAIN_IMAGE:-qed-nano-train}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Build flag
if [[ "${1:-}" == "--build" ]]; then
  shift
  echo "Building ${IMAGE} (this takes ~10 min on first build due to flash-attn)..."
  docker build -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile.train" "${SCRIPT_DIR}"
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image '${IMAGE}' not found. Run: bash train_rl.sh --build" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/outputs"

TOTAL_GPUS=$(( ${ACTOR_GPUS:-2} + ${TRAINER_GPUS:-2} ))
echo "Starting RL training (image: ${IMAGE}, ${TOTAL_GPUS} GPUs)"

docker run --rm --gpus all --ipc=host --shm-size 32g \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -v "${SCRIPT_DIR}/conf/rl_cvl.yaml:/opt/qed-nano/training/conf/rl_cvl.yaml:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo" \
  -e PIPELINERL_ROOT="/opt/qed-nano/training" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}" \
  -e MODEL_REVISION="${MODEL_REVISION:-main}" \
  -e ACTOR_GPUS="${ACTOR_GPUS:-2}" \
  -e TRAINER_GPUS="${TRAINER_GPUS:-2}" \
  -e SEQ_LENGTH="${SEQ_LENGTH:-8192}" \
  -e GRADER_MODEL="${GRADER_MODEL:-gpt-4o-mini}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-qed-nano-rl}" \
  -e OUTPUT_DIR="${OUTPUT_DIR:-}" \
  "${IMAGE}" \
  python train_rl.py "$@"
