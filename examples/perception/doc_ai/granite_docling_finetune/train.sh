#!/bin/bash
set -e

# Default values
TRAIN_DATA="${TRAIN_DATA:-ds4sd/docling-dpbench}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/granite_docling_sft}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-1e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
LORA_R="${LORA_R:-16}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --lora-r)
            LORA_R="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Granite-Docling fine-tuning..."
echo "Train data: $TRAIN_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Docker container with GPU support
docker run --rm \
    --gpus all \
    -v "$(pwd):/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e TRAIN_DATA="$TRAIN_DATA" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e GRAD_ACCUM="$GRAD_ACCUM" \
    -e NUM_EPOCHS="$NUM_EPOCHS" \
    -e LR="$LR" \
    -e MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    -e LORA_R="$LORA_R" \
    -e MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}" \
    granite-docling-finetune \
    python train.py

echo ""
echo "Training complete! LoRA adapters saved to: $OUTPUT_DIR/lora_adapters"
