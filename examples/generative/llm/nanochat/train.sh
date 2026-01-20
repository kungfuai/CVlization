#!/bin/bash
# Train nanochat model on a single GPU
# Supports all 4 training stages: base, mid, sft, rl
#
# Usage:
#   bash train.sh [mode] [options]
#
# Modes:
#   base  - Base pretraining (default)
#   mid   - Midtraining on curated data
#   sft   - Supervised fine-tuning for chat
#   rl    - Reinforcement learning
#   all   - Run full pipeline (base -> mid -> sft -> rl)
#
# Examples:
#   bash train.sh base --depth=8 --num_iterations=100
#   bash train.sh sft --depth=8
#   bash train.sh all --depth=4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse mode (default to all stages except RL)
MODE="${1:-all}"
shift || true  # Remove mode from args, continue if no args left

# Validate mode
case "$MODE" in
    base|mid|sft|rl|all|full)
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Valid modes: base, mid, sft, rl, all, full"
        echo "  all  = base -> mid -> sft (default, no RL)"
        echo "  full = base -> mid -> sft -> rl (includes RL)"
        exit 1
        ;;
esac

# Create output directories
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/outputs"

echo "==================================="
echo "nanochat Training - Mode: $MODE"
echo "==================================="
echo ""

# Function to run a training stage
run_stage() {
    local stage=$1
    shift
    local script_name=$1
    shift
    local description=$1
    shift

    echo ""
    echo ">>> Starting: $description"
    echo ""

    REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

    docker run --rm --gpus all \
        --workdir /workspace \
        -e PYTHONUNBUFFERED=1 \
        -v "$SCRIPT_DIR:/workspace" \
        -v "${CVL_NANOCHAT_CACHE:-$HOME/.cache/nanochat}:/root/.cache/nanochat" \
        nanochat \
        bash -c "
            cd /workspace/nanochat

            # Build tokenizer if needed (only for base stage)
            if [ '$stage' = 'base' ]; then
                echo '=== Building tokenizer ==='
                uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
                echo ''

                # Download training data
                echo '=== Downloading training data ==='
                uv run python -m nanochat.dataset -n 10
                echo ''

                # Train tokenizer
                echo '=== Training tokenizer ==='
                uv run python -m scripts.tok_train --max-chars 500000000
                echo ''
            fi

            # Run the training script
            echo '=== Running $description ==='
            uv run python -m scripts.$script_name $@

            echo ''
            echo '$description complete!'
        "
}

# Run training based on mode
case "$MODE" in
    base)
        run_stage base base_train "Base Pretraining" --depth=8 --device_batch_size=4 --num_iterations=100 "$@"
        ;;
    mid)
        run_stage mid mid_train "Midtraining" --depth=8 --device_batch_size=4 "$@"
        ;;
    sft)
        run_stage sft chat_sft "Supervised Fine-Tuning" --depth=8 --device_batch_size=4 "$@"
        ;;
    rl)
        run_stage rl chat_rl "Reinforcement Learning" --depth=8 --device_batch_size=4 "$@"
        ;;
    all)
        echo "Running training pipeline (base -> mid -> sft)..."
        run_stage base base_train "Base Pretraining" --depth=8 --device_batch_size=4 --num_iterations=100 "$@"
        run_stage mid mid_train "Midtraining" --depth=8 --device_batch_size=4 "$@"
        run_stage sft chat_sft "Supervised Fine-Tuning" --depth=8 --device_batch_size=4 "$@"
        echo ""
        echo "Training pipeline complete! (use 'full' mode to include RL)"
        ;;
    full)
        echo "Running full training pipeline with RL..."
        run_stage base base_train "Base Pretraining" --depth=8 --device_batch_size=4 --num_iterations=100 "$@"
        run_stage mid mid_train "Midtraining" --depth=8 --device_batch_size=4 "$@"
        run_stage sft chat_sft "Supervised Fine-Tuning" --depth=8 --device_batch_size=4 "$@"
        run_stage rl chat_rl "Reinforcement Learning" --depth=8 --device_batch_size=4 "$@"
        echo ""
        echo "Full pipeline with RL complete!"
        ;;
esac

echo ""
echo "Training finished! Check outputs/ directory for checkpoints."
