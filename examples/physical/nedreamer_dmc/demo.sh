#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cvl-nedreamer-dmc}"
TAG="${TAG:-latest}"

TASK="${TASK:-dmc_walker_walk}"
STEPS="${STEPS:-200000}"

echo "=== NE-Dreamer Demo: ${TASK}, ${STEPS} steps ==="
echo "Expected time: ~20 min on A10/A100"

# Clean previous outputs (may be root-owned from docker)
docker run --rm \
  --mount "type=bind,src=${SCRIPT_DIR}/outputs,dst=/outputs" \
  alpine rm -rf /outputs/logdir 2>/dev/null || rm -rf "${SCRIPT_DIR}/outputs/logdir"
mkdir -p "${SCRIPT_DIR}/outputs"

# Train with eval video logging enabled
STEPS="${STEPS}" TASK="${TASK}" \
  bash "${SCRIPT_DIR}/train.sh" \
    trainer.eval_every=10000 \
    trainer.eval_video_every=1 \
    trainer.video_pred_log=False \
    model.compile=False \
    trainer.s3_bucket=null

echo ""
echo "=== Training complete. Generating plot... ==="

# Plot training curve (run inside container for matplotlib)
GPU_FLAG="--gpus=all"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_FLAG="--gpus=\"device=${CUDA_VISIBLE_DEVICES}\""
fi

eval docker run --rm ${GPU_FLAG} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  "${IMAGE_NAME}:${TAG}" \
  python plot_training.py \
    --logdir outputs/logdir \
    --output outputs/training_curve.png

# Fix ownership
docker run --rm \
  --mount "type=bind,src=${SCRIPT_DIR}/outputs,dst=/outputs" \
  alpine chown -R "$(id -u):$(id -g)" /outputs 2>/dev/null || true

echo ""
echo "=== Demo outputs ==="
echo "Training curve: outputs/training_curve.png"
echo "Eval videos:    outputs/logdir/eval_step_*.mp4"
ls -lh "${SCRIPT_DIR}"/outputs/logdir/eval_step_*.mp4 2>/dev/null || echo "(no videos found)"
ls -lh "${SCRIPT_DIR}"/outputs/training_curve.png 2>/dev/null || echo "(no plot found)"
