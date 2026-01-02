#!/bin/bash

# Training script for modded-nanogpt with Docker

# Default values
NUM_GPUS=${NUM_GPUS:-8}
GPU_ID=${GPU_ID:-}  # Empty means use all GPUs (0,1,2,...NUM_GPUS-1)
DATA_DIR=${DATA_DIR:-"./data"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
LOG_DIR=${LOG_DIR:-"./logs"}
WANDB_MODE=${WANDB_MODE:-"disabled"}
WANDB_PROJECT=${WANDB_PROJECT:-"modded_nanogpt_a10"}
USE_COMPILE=${USE_COMPILE:-"1"}  # Default: enabled (1), disabled (0)
# Memory optimization: smaller sequence lengths for A10 GPU (23GB)
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-$((48*1024))}  # Default: 48K tokens
VAL_SEQ_LEN=${VAL_SEQ_LEN:-$((4*64*1024))}    # Default: 256K tokens
MAX_BATCH_SPAN_MULTIPLIER=${MAX_BATCH_SPAN_MULTIPLIER:-4}  # Default: 4x batch size for BOS alignment
TRAIN_LOSS_EVERY=${TRAIN_LOSS_EVERY:-0}  # Default: 0 (disabled)
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-125}  # Default: 125 (val loss every 125 steps)
VAL_TOKENS=${VAL_TOKENS:-10485760}  # Default: 10M tokens for validation
NUM_ITERATIONS=${NUM_ITERATIONS:-3500}  # Default: 3500 training steps
# Learning rates
ADAM_LR=${ADAM_LR:-0.008}  # Default: 0.008 for DistAdam optimizer
MUON_LR=${MUON_LR:-0.05}   # Default: 0.05 for Muon optimizer

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --wandb)
      WANDB_MODE="online"
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --train-seq-len)
      TRAIN_SEQ_LEN="$2"
      shift 2
      ;;
    --val-seq-len)
      VAL_SEQ_LEN="$2"
      shift 2
      ;;
    --no-torch-compile)
      USE_COMPILE="0"
      shift
      ;;
    --max-batch-span-multiplier)
      MAX_BATCH_SPAN_MULTIPLIER="$2"
      shift 2
      ;;
    --train-loss-every)
      TRAIN_LOSS_EVERY="$2"
      shift 2
      ;;
    --val-loss-every)
      VAL_LOSS_EVERY="$2"
      shift 2
      ;;
    --val-tokens)
      VAL_TOKENS="$2"
      shift 2
      ;;
    --num-iterations)
      NUM_ITERATIONS="$2"
      shift 2
      ;;
    --adam-lr)
      ADAM_LR="$2"
      shift 2
      ;;
    --muon-lr)
      MUON_LR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./train.sh [OPTIONS]"
      echo ""
      echo "GPU options:"
      echo "  --num-gpus N           Number of GPUs to use (default: 8)"
      echo "  --gpu-id N             Specific GPU ID to use (overrides --num-gpus)"
      echo ""
      echo "Memory optimization options:"
      echo "  --train-seq-len N      Training sequence length in tokens (default: 49152)"
      echo "  --val-seq-len N        Validation sequence length in tokens (default: 262144)"
      echo "  --val-tokens N         Total validation tokens (default: 10485760)"
      echo "  --max-batch-span-multiplier N   Multiplier for BOS alignment (default: 4)"
      echo ""
      echo "Training options:"
      echo "  --num-iterations N     Number of training steps (default: 3500)"
      echo "  --adam-lr N            DistAdam learning rate (default: 0.008)"
      echo "  --muon-lr N            Muon learning rate (default: 0.05)"
      echo "  --no-torch-compile     Disable torch.compile (faster startup, slower training)"
      echo ""
      echo "Logging options:"
      echo "  --train-loss-every N   Print train loss every N steps (default: 0=disabled)"
      echo "  --val-loss-every N     Evaluate val loss every N steps (default: 125)"
      echo "  --wandb                Enable Weights & Biases logging"
      echo ""
      echo "Example for single GPU (24GB):"
      echo "  ./train.sh --gpu-id 0 --num-gpus 1 --train-seq-len 4096 --val-seq-len 16384 \\"
      echo "             --val-tokens 65536 --max-batch-span-multiplier 32 --train-loss-every 10"
      echo ""
      echo "Example for 8xH100 (full training):"
      echo "  ./train.sh --num-gpus 8 --wandb"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

echo "Starting modded-nanogpt training with:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Train sequence length: $TRAIN_SEQ_LEN tokens (also sets effective batch size)"
echo "  - Validation sequence length: $VAL_SEQ_LEN tokens"
echo "  - Data directory: $DATA_DIR"
echo "  - Checkpoint directory: $CHECKPOINT_DIR"
echo "  - Log directory: $LOG_DIR"
echo "  - Weights & Biases: $WANDB_MODE"
echo "  - Torch compile: $([ "$USE_COMPILE" = "1" ] && echo "enabled" || echo "disabled")"
echo "  - Max batch span multiplier: $MAX_BATCH_SPAN_MULTIPLIER"
echo "  - Train loss printing: $([ "$TRAIN_LOSS_EVERY" = "0" ] && echo "disabled" || echo "every $TRAIN_LOSS_EVERY steps")"
echo "  - Val loss printing: every $VAL_LOSS_EVERY steps"
echo "  - Val tokens: $VAL_TOKENS"
echo "  - Training iterations: $NUM_ITERATIONS steps"
echo "  - DistAdam learning rate: $ADAM_LR"
echo "  - Muon learning rate: $MUON_LR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HuggingFace cache for FineWeb data (centralized on host)
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Run the Docker container with unbuffered output
stdbuf -o0 -e0 docker run --rm \
  --gpus all \
  --ipc=host \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_MODE="$WANDB_MODE" \
  -e WANDB_PROJECT="$WANDB_PROJECT" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e CUDA_VISIBLE_DEVICES="${GPU_ID:-$(seq -s, 0 $((NUM_GPUS-1)))}" \
  -e TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
  -e VAL_SEQ_LEN="$VAL_SEQ_LEN" \
  -e USE_COMPILE="$USE_COMPILE" \
  -e MAX_BATCH_SPAN_MULTIPLIER="$MAX_BATCH_SPAN_MULTIPLIER" \
  -e TRAIN_LOSS_EVERY="$TRAIN_LOSS_EVERY" \
  -e VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  -e VAL_TOKENS="$VAL_TOKENS" \
  -e NUM_ITERATIONS="$NUM_ITERATIONS" \
  -e ADAM_LR="$ADAM_LR" \
  -e MUON_LR="$MUON_LR" \
  modded-nanogpt \
  bash -c "
    # Download the training data if it doesn't exist (cached in host ~/.cache/huggingface)
    if [ ! -f /workspace/data/fineweb10B/fineweb_val_000000.bin ]; then
      echo 'Downloading FineWeb training data from HuggingFace (first 10 shards = 1B tokens)...'
      python data/cached_fineweb10B.py 10
    else
      echo 'Using cached FineWeb data (HuggingFace cache)'
    fi

    # Run the training
    echo 'Starting training...'
    # Note: train_gpt.py has its own batch size settings internally
    # To modify batch size, you would need to edit train_gpt.py directly
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py
  "