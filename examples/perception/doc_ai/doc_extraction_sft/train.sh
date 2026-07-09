#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-doc_extraction_sft}"
CACHE_DIR="${CVL_HF_CACHE:-$HOME/.cache/huggingface}"
DATA_DIR="${CVL_DATA_DIR:-/data}"
DOCKER_GPUS="${CVL_DOCKER_GPUS:-all}"

for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        cat <<'EOF'
Train a document extraction SFT model.

Usage:
  CVL_IMAGE=doc_extraction_sft_modern CVL_GPUS=0 \
    cvl run doc_extraction_sft train --config config.yaml --data-files /path/to/train.jsonl
  CVL_IMAGE=doc_extraction_sft_modern CVL_GPUS=0 ./train.sh --config config.yaml --data-files /path/to/train.jsonl

Dataset config:
  dataset:
    path: "json"
    data_files: "$DOC_EXTRACTION_SFT_TRAIN_JSONL"  # or pass --data-files at runtime
    split: "train"
    target_column: "target_json"
    target_sources:
      - "ground_truth"
      - "form_identification"
    split_strategy: "document_id"
    eval_split_ratio: 0.1

Model and training config:
  model:
    name: "Qwen/Qwen3-8B"
    max_seq_length: 32768
    max_target_length: 8192
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: "all-linear"
  training:
    output_dir: "outputs/qwen3_8b"
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 0.0002
    num_train_epochs: 1
    loss: "assistant_only"

The dataset path can live in the config, be supplied with --data-files, or come
from DOC_EXTRACTION_SFT_TRAIN_JSONL for backwards-compatible shared configs.
EOF
        exit 0
    fi
done

if [ -n "${CVL_GPUS:-}" ]; then
    DOCKER_GPUS="\"device=${CVL_GPUS}\""
fi

mkdir -p "$SCRIPT_DIR/outputs" "$CACHE_DIR"

echo "=== Document Extraction SFT ==="
echo "Image: $IMG"
echo "HF cache: $CACHE_DIR"
echo "GPUs: $DOCKER_GPUS"

if [ "${CVL_NO_DOCKER:-}" = "1" ]; then
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    python3 train.py "$@"
    echo "Training complete."
    exit 0
fi

docker run --rm --gpus "$DOCKER_GPUS" --shm-size 16G \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${CACHE_DIR},dst=/root/.cache/huggingface" \
    $(if [ -d "$DATA_DIR" ]; then printf '%s\n' --mount "type=bind,src=${DATA_DIR},dst=${DATA_DIR}"; fi) \
    --env "PYTHONPATH=/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    ${DOC_EXTRACTION_SFT_TRAIN_JSONL:+--env DOC_EXTRACTION_SFT_TRAIN_JSONL="$DOC_EXTRACTION_SFT_TRAIN_JSONL"} \
    ${PYTORCH_CUDA_ALLOC_CONF:+--env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
    ${PYTORCH_ALLOC_CONF:+--env PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"} \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    "$IMG" python3 train.py "$@"

echo "Training complete."
