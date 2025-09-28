#!/bin/bash

# Training script for modded-nanogpt with Docker

# Default values
NUM_GPUS=${NUM_GPUS:-8}
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
NUM_ITERATIONS=${NUM_ITERATIONS:-3500}  # Default: 3500 training steps

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-gpus)
      NUM_GPUS="$2"
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
    --num-iterations)
      NUM_ITERATIONS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./train.sh [--num-gpus N] [--wandb] [--data-dir DIR] [--train-seq-len N] [--val-seq-len N] [--no-torch-compile] [--max-batch-span-multiplier N] [--train-loss-every N] [--num-iterations N]"
      echo ""
      echo "Memory optimization options:"
      echo "  --train-seq-len N      Training sequence length in tokens (default: 49152)"
      echo "  --val-seq-len N        Validation sequence length in tokens (default: 262144)"
      echo ""
      echo "Performance options:"
      echo "  --no-torch-compile     Disable torch.compile for faster startup (slower training)"
      echo "  --max-batch-span-multiplier N   Multiplier for max_batch_span when align_to_bos=True (default: 4)"
      echo "  --train-loss-every N   Print training loss every N steps (default: 0=disabled)"
      echo "  --num-iterations N     Number of training steps to run (default: 3500)"
      echo ""
      echo "Example for A10 GPU (23GB memory):"
      echo "  ./train.sh --train-seq-len 16384 --val-seq-len 65536"
      echo ""
      echo "Example to disable torch.compile:"
      echo "  ./train.sh --no-torch-compile"
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
echo "  - Training iterations: $NUM_ITERATIONS steps"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Docker container
docker run -it --rm \
  --gpus all \
  --ipc=host \
  -v "$SCRIPT_DIR:/workspace" \
  -e WANDB_MODE="$WANDB_MODE" \
  -e WANDB_PROJECT="$WANDB_PROJECT" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))" \
  -e TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
  -e VAL_SEQ_LEN="$VAL_SEQ_LEN" \
  -e USE_COMPILE="$USE_COMPILE" \
  -e MAX_BATCH_SPAN_MULTIPLIER="$MAX_BATCH_SPAN_MULTIPLIER" \
  -e TRAIN_LOSS_EVERY="$TRAIN_LOSS_EVERY" \
  -e NUM_ITERATIONS="$NUM_ITERATIONS" \
  modded-nanogpt \
  bash -c "
    # First, download the training data if it doesn't exist
    if [ ! -f /workspace/data/fineweb_train_00000000.bin ]; then
      echo 'Downloading FineWeb training data (first 10 shards = 1B tokens)...'
      python data/cached_fineweb10B.py 10
    fi

    # Run the training
    echo 'Starting training...'
    # Note: train_gpt.py has its own batch size settings internally
    # To modify batch size, you would need to edit train_gpt.py directly
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py
  "