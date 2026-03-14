#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cvl-nedreamer-dmc}"
TAG="${TAG:-latest}"

mkdir -p "${SCRIPT_DIR}/outputs"

GPU_FLAG="--gpus=all"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_FLAG="--gpus='\"device=${CUDA_VISIBLE_DEVICES}\"'"
fi

eval docker run --rm ${GPU_FLAG} \
  --workdir /workspace/nedreamer \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" \
  --env "WANDB_MODE=${WANDB_MODE:-disabled}" \
  --env "MUJOCO_GL=egl" \
  "${IMAGE_NAME}:${TAG}" \
  python train.py hydra.run.dir=/workspace/outputs/hydra \
    model=size12M \
    model.rep_loss=ne_dreamer \
    env.task="${TASK:-dmc_walker_walk}" \
    env.steps="${STEPS:-10000}" \
    env.env_num="${ENV_NUM:-4}" \
    env.eval_episode_num="${EVAL_EPISODES:-2}" \
    seed="${SEED:-0}" \
    device="${DEVICE:-cuda:0}" \
    logdir=/workspace/outputs/logdir \
    "$@"

# Fix ownership of outputs created by root inside container
if [ -d "${SCRIPT_DIR}/outputs" ]; then
  docker run --rm \
    --mount "type=bind,src=${SCRIPT_DIR}/outputs,dst=/outputs" \
    alpine chown -R "$(id -u):$(id -g)" /outputs 2>/dev/null || true
fi
