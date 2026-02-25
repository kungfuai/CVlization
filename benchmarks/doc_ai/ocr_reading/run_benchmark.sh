#!/bin/bash
# OCR Reading Benchmark - Main Entry Point
#
# Usage:
#   ./run_benchmark.sh [options]
#
# Options:
#   --model     <name>   Model adapter to use (default: qwen25_vl_7b)
#   --split     <split>  Dataset split (default: test)
#   --output-dir <dir>   Results directory (default: results/)
#   --limit     <n>      Max samples (default: 0 = all)
#   --skip-build         Skip Docker image build
#   --skip-download      Skip dataset download (use existing shards)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MODEL="qwen25_vl_7b"
SPLIT="train"
OUTPUT_DIR="results"
LIMIT=0
SKIP_BUILD=false
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --split)        SPLIT="$2"; shift 2 ;;  # 'train' only for this dataset
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=true; shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ADAPTER="./adapters/${MODEL}.sh"

if [ ! -f "$ADAPTER" ]; then
    echo "Error: Adapter not found: $ADAPTER"
    echo "Available adapters:"
    ls adapters/*.sh 2>/dev/null | sed 's|adapters/||; s|\.sh||'
    exit 1
fi

# Work directory: use CVlization cache pattern
CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
WORK_DIR="${CACHE_HOME}/cvlization/data/ocr_reading/sroie/${SPLIT}"
PREDS_DIR="${OUTPUT_DIR}/predictions"
METRICS_FILE="${OUTPUT_DIR}/metrics.json"

mkdir -p "$PREDS_DIR" "$OUTPUT_DIR"

echo "========================================"
echo "OCR Reading Benchmark"
echo "Model:      $MODEL"
echo "Split:      $SPLIT"
echo "Work dir:   $WORK_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Step 1: Build Docker image
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "Step 1/4: Building Docker image..."
    bash "$ADAPTER" build
else
    echo "Step 1/4: Skipping Docker build"
fi

# Step 2: Download and build TAR shards
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "Step 2/4: Downloading SROIE dataset and building shards..."
    python3 dataset_builder.py \
        --split "$SPLIT" \
        --output-dir "$WORK_DIR"
else
    echo "Step 2/4: Skipping dataset download"
fi

# Check shards exist
SHARD_COUNT=$(ls "$WORK_DIR"/shard-*.tar 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "Error: No shards found in $WORK_DIR"
    exit 1
fi
echo "Found $SHARD_COUNT shards in $WORK_DIR"

# Step 3: Run inference + scoring
PRED_CSV="$PREDS_DIR/${MODEL}_predictions.csv"
echo ""
echo "Step 3/4: Running inference and scoring..."
bash "$ADAPTER" \
    --shards "$WORK_DIR" \
    --output "$PRED_CSV" \
    --limit "$LIMIT"

# Step 4: Aggregate metrics
echo ""
echo "Step 4/4: Aggregating metrics..."
python3 evaluate.py \
    --pred "$PRED_CSV" \
    --output "$METRICS_FILE" \
    --task "reading"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Predictions: $PRED_CSV"
echo "Metrics:     $METRICS_FILE"
echo "========================================"
cat "$METRICS_FILE"
