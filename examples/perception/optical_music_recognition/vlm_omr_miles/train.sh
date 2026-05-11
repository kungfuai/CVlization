#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SFT_DIR="$REPO_ROOT/examples/perception/optical_music_recognition/vlm_omr_sft"
IMG="${CVL_IMAGE:-cvlization/vlm-omr-miles:latest}"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "${SCRIPT_DIR}/outputs"

# Mount the SFT directory read-only so the merged model + mxc2.py are accessible.
SFT_MOUNT=""
if [ -d "$SFT_DIR/outputs/tamqjf4k_merged" ]; then
  SFT_MOUNT="--mount type=bind,src=${SFT_DIR},dst=/sft_workspace,readonly"
  echo "Mounting SFT dir: ${SFT_DIR} -> /sft_workspace (readonly)"
fi

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  $SFT_MOUNT \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/workspace:/cvlization_repo:/root/Megatron-LM" \
  --env "PYTHONUNBUFFERED=1" \
  ${WANDB_API_KEY:+--env "WANDB_API_KEY=${WANDB_API_KEY}"} \
  --shm-size=16g \
  --ipc=host \
  "$IMG" python3 train.py "$@"
