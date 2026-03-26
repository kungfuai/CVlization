#!/bin/bash
# Adapter: Qwen2.5-VL-7B-Instruct on OCR Reading Benchmark
#
# Usage:
#   ./adapters/qwen25_vl_7b.sh build
#   ./adapters/qwen25_vl_7b.sh --shards <shards_dir> --output <predictions.csv> [options]
#
# Options:
#   --shards   <dir>   Directory containing shard-*.tar files (required)
#   --output   <file>  Output CSV path (default: predictions.csv)
#   --task     <type>  Task type (default: reading)
#   --output-type <t>  Output type (default: "[lines, box]")
#   --gpus     <n>     Number of GPUs (default: 1)
#   --limit    <n>     Max samples to evaluate (default: 0 = all)
#   --skip-score       Skip scoring step (inference only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
SHARDS_DIR=""
OUTPUT_CSV="predictions.csv"
TASK_TYPE="reading"
OUTPUT_TYPE="[lines, box]"
NUM_GPUS=1
LIMIT=0
SKIP_SCORE=false
IMAGE_NAME="cvlization/ocr-eval-qwen25:latest"

# Parse arguments
SUBCMD="$1"
if [ "$SUBCMD" = "build" ]; then
    shift
    BUILD_ONLY=true
else
    BUILD_ONLY=false
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards)     SHARDS_DIR="$2"; shift 2 ;;
        --output)     OUTPUT_CSV="$2"; shift 2 ;;
        --task)       TASK_TYPE="$2"; shift 2 ;;
        --output-type) OUTPUT_TYPE="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --limit)      LIMIT="$2"; shift 2 ;;
        --skip-score) SKIP_SCORE=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- BUILD subcommand ------------------------------------------------
if [ "$BUILD_ONLY" = true ]; then
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" -f - "$BENCHMARK_DIR" << 'DOCKERFILE'
FROM vllm/vllm-openai:v0.9.0

WORKDIR /workspace

# Install scoring dependencies not in the base vllm image
RUN pip install --no-cache-dir \
    Levenshtein \
    jiwer \
    json-repair \
    scipy \
    tabulate \
    addict \
    easydict \
    einops \
    timm

# Copy vendored scripts (flat layout — all files directly in vendor/)
COPY vendor /workspace/vendor

ENV PYTHONPATH=/workspace/vendor
DOCKERFILE
    echo "Build complete: $IMAGE_NAME"
    exit 0
fi

# ---- RUN subcommand --------------------------------------------------
if [ -z "$SHARDS_DIR" ]; then
    echo "Error: --shards <dir> is required"
    exit 1
fi

SHARDS_ABS="$(realpath "$SHARDS_DIR")"
OUTPUT_ABS="$(realpath -m "$OUTPUT_CSV")"
OUTPUT_DIR="$(dirname "$OUTPUT_ABS")"

mkdir -p "$OUTPUT_DIR"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE_DIR"

# Find all shard tarballs, sorted
mapfile -t SHARDS_LIST < <(find "$SHARDS_ABS" -maxdepth 1 -name 'shard-*.tar' | sort)
NUM_SHARDS="${#SHARDS_LIST[@]}"

if [ "$NUM_SHARDS" -eq 0 ]; then
    echo "Error: No shard-*.tar files found in $SHARDS_ABS"
    exit 1
fi

echo "Found $NUM_SHARDS shards in $SHARDS_ABS"
echo "Task: $TASK_TYPE / $OUTPUT_TYPE"
echo "Output: $OUTPUT_ABS"

# We run inference shard-by-shard and concatenate results, or pass
# all shards sequentially. The vendored run_evaluation.py takes a single
# --shard-path at a time, so we iterate and merge CSVs.

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

MERGED_PRED_CSV="$TEMP_DIR/all_predictions.csv"
HEADER_WRITTEN=false

for SHARD in "${SHARDS_LIST[@]}"; do
    SHARD_NAME="$(basename "$SHARD" .tar)"
    SHARD_PRED="$TEMP_DIR/${SHARD_NAME}-pred.csv"

    echo "--- Inferring $SHARD_NAME ---"

    docker run --runtime nvidia --rm \
        --gpus "device=0" \
        -v "$SHARDS_ABS:/shards:ro" \
        -v "$TEMP_DIR:/tmp/ocr_output" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -e PYTHONPATH=/workspace/vendor \
        -e HF_HOME=/root/.cache/huggingface \
        "$IMAGE_NAME" \
        python /workspace/vendor/run_evaluation.py \
            --shard-path "/shards/$(basename "$SHARD")" \
            --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
            --task-types "$TASK_TYPE" \
            --output-types "$OUTPUT_TYPE" \
            --limit "$LIMIT" \
            --num-workers "$NUM_GPUS" \
            --csv-output "/tmp/ocr_output/${SHARD_NAME}-pred.csv"

    # Merge CSV (skip header after first shard)
    if [ -f "$SHARD_PRED" ]; then
        if [ "$HEADER_WRITTEN" = false ]; then
            cat "$SHARD_PRED" >> "$MERGED_PRED_CSV"
            HEADER_WRITTEN=true
        else
            tail -n +2 "$SHARD_PRED" >> "$MERGED_PRED_CSV"
        fi
    fi
done

if [ "$SKIP_SCORE" = true ]; then
    cp "$MERGED_PRED_CSV" "$OUTPUT_ABS"
    echo "Predictions (no scoring) saved to $OUTPUT_ABS"
    exit 0
fi

# ---- SCORING step ---------------------------------------------------
echo "--- Scoring predictions ---"
SCORED_CSV="$TEMP_DIR/scored.csv"
cp "$MERGED_PRED_CSV" "$SCORED_CSV"

docker run --rm \
    -v "$TEMP_DIR:/tmp/ocr_output" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -e PYTHONPATH=/workspace/vendor \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE_NAME" \
    python /workspace/vendor/score_lines_reading.py \
        /tmp/ocr_output/scored.csv \
        --overwrite

cp "$SCORED_CSV" "$OUTPUT_ABS"
echo "Scored predictions saved to $OUTPUT_ABS"
