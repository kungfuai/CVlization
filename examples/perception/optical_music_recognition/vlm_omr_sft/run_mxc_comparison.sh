#!/usr/bin/env bash
# Run all 4 MXC model comparison experiments sequentially.
# Each runs 3 epochs with r=32, MXC targets, 10 inference examples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================"
echo "MXC Model Comparison — 4 models × 3 epochs"
echo "============================================"

# 1. Gemma-3 4B
echo ""
echo "[1/4] Gemma-3 4B MXC ..."
CVL_CONTAINER_NAME=gemma3_4b_mxc bash train.sh --config config_gemma3_4b_mxc.yaml \
  2>&1 | tee /tmp/gemma3_4b_mxc.log
echo "[1/4] Gemma-3 4B MXC — done"

# 2. Qwen3-VL 8B
echo ""
echo "[2/4] Qwen3-VL 8B MXC ..."
CVL_CONTAINER_NAME=qwen3vl_8b_mxc bash train.sh --config config_qwen3vl_8b_mxc.yaml \
  2>&1 | tee /tmp/qwen3vl_8b_mxc.log
echo "[2/4] Qwen3-VL 8B MXC — done"

# 3. Qwen3-VL 32B
echo ""
echo "[3/4] Qwen3-VL 32B MXC ..."
CVL_CONTAINER_NAME=qwen3vl_32b_mxc bash train.sh --config config_qwen3vl_32b_mxc.yaml \
  2>&1 | tee /tmp/qwen3vl_32b_mxc.log
echo "[3/4] Qwen3-VL 32B MXC — done"

# 4. DeepSeek-OCR-2 (uses train_deepseek_ocr2.py, not train.py)
echo ""
echo "[4/4] DeepSeek-OCR-2 MXC ..."
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/vlm-omr-sft:latest}"
docker run --rm --gpus=all \
  --name deepseek_ocr2_mxc \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  ${WANDB_API_KEY:+--env "WANDB_API_KEY=${WANDB_API_KEY}"} \
  ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"} \
  --shm-size=16g \
  "$IMG" python3 train_deepseek_ocr2.py --config config_deepseek_ocr2_mxc.yaml \
  2>&1 | tee /tmp/deepseek_ocr2_mxc.log
echo "[4/4] DeepSeek-OCR-2 MXC — done"

echo ""
echo "============================================"
echo "All 4 runs complete. Check WandB for results."
echo "Run: python eval_mxc.py --latest"
echo "============================================"
