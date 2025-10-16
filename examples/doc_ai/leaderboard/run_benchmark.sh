#!/bin/bash
set -e

# Simple benchmark runner for doc AI models
# Usage: ./run_benchmark.sh [models...]
# Example: ./run_benchmark.sh moondream2 granite-docling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load config
CONFIG_FILE="config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found"
    exit 1
fi

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Save run metadata
echo "Run started: $(date)" > "$RESULTS_DIR/run_info.txt"
echo "Models: $@" >> "$RESULTS_DIR/run_info.txt"

# Read test images from config
readarray -t TEST_IMAGES < <(jq -r '.test_images[]' "$CONFIG_FILE")

# Determine which models to run
if [ $# -eq 0 ]; then
    # Run all models from config
    readarray -t MODELS < <(jq -r '.models | keys[]' "$CONFIG_FILE")
else
    # Run specified models
    MODELS=("$@")
fi

echo "================================"
echo "Doc AI Leaderboard Benchmark"
echo "================================"
echo "Timestamp: $TIMESTAMP"
echo "Models: ${MODELS[@]}"
echo "Test images: ${#TEST_IMAGES[@]}"
echo "Results: $RESULTS_DIR"
echo ""

# CSV header
echo "model,image,time_seconds,status" > "$RESULTS_DIR/benchmark.csv"

# Run benchmark
for image_path in "${TEST_IMAGES[@]}"; do
    # Resolve relative path
    if [[ "$image_path" != /* ]]; then
        image_path="$SCRIPT_DIR/$image_path"
    fi

    if [ ! -f "$image_path" ]; then
        echo "Warning: Image not found: $image_path"
        continue
    fi

    image_name=$(basename "$image_path")
    echo "Processing: $image_name"
    echo "----------------------------"

    for model in "${MODELS[@]}"; do
        echo "  [$model]"

        # Get model script and args from config
        script=$(jq -r ".models.\"$model\".script" "$CONFIG_FILE")
        args=$(jq -r ".models.\"$model\".args" "$CONFIG_FILE")

        if [ "$script" = "null" ]; then
            echo "    ERROR: Model '$model' not found in config"
            echo "$model,$image_name,0,error_not_configured" >> "$RESULTS_DIR/benchmark.csv"
            continue
        fi

        # Create model output directory
        model_output_dir="$RESULTS_DIR/$model"
        mkdir -p "$model_output_dir"

        # Resolve script path
        script_path="$SCRIPT_DIR/$script"
        if [ ! -f "$script_path" ]; then
            echo "    ERROR: Script not found: $script_path"
            echo "$model,$image_name,0,error_script_missing" >> "$RESULTS_DIR/benchmark.csv"
            continue
        fi

        # Run inference and time it
        output_file="$model_output_dir/${image_name%.*}_output.txt"
        log_file="$model_output_dir/${image_name%.*}_log.txt"

        start_time=$(date +%s.%N)
        status="success"

        # Run the model's predict script
        if bash "$script_path" "$image_path" --output "$output_file" $args > "$log_file" 2>&1; then
            end_time=$(date +%s.%N)
            duration=$(echo "$end_time - $start_time" | bc)
            echo "    ✓ Time: ${duration}s"
        else
            end_time=$(date +%s.%N)
            duration=$(echo "$end_time - $start_time" | bc)
            status="failed"
            echo "    ✗ Failed (${duration}s)"
        fi

        # Record results
        echo "$model,$image_name,$duration,$status" >> "$RESULTS_DIR/benchmark.csv"
    done

    echo ""
done

echo "================================"
echo "Benchmark Complete!"
echo "================================"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Generate summary
echo "Summary:"
echo "--------"
for model in "${MODELS[@]}"; do
    total_time=$(awk -F',' -v m="$model" '$1==m && $4=="success" {sum+=$3; count++} END {print sum}' "$RESULTS_DIR/benchmark.csv")
    success_count=$(awk -F',' -v m="$model" '$1==m && $4=="success" {count++} END {print count}' "$RESULTS_DIR/benchmark.csv")
    total_count=$(awk -F',' -v m="$model" '$1==m {count++} END {print count}' "$RESULTS_DIR/benchmark.csv")

    if [ -n "$success_count" ] && [ "$success_count" -gt 0 ]; then
        avg_time=$(echo "scale=2; $total_time / $success_count" | bc)
        echo "  $model: ${success_count}/${total_count} succeeded, avg time: ${avg_time}s"
    else
        echo "  $model: 0/${total_count} succeeded"
    fi
done

echo ""
echo "View full results:"
echo "  cat $RESULTS_DIR/benchmark.csv"
echo "  ls $RESULTS_DIR/*/"
