#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=("$@")

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=(
        config_qwen3_8b.yaml
        config_phi4_14b.yaml
        config_qwen3_14b.yaml
    )
fi

mkdir -p "$SCRIPT_DIR/outputs/sweep_logs"

export CVL_IMAGE="${CVL_IMAGE:-doc_extraction_sft_modern}"

for config in "${CONFIGS[@]}"; do
    name="$(basename "$config" .yaml)"
    log="$SCRIPT_DIR/outputs/sweep_logs/${name}.log"
    echo "=== Running $config ==="
    echo "Log: $log"
    "$SCRIPT_DIR/train.sh" --config "$config" 2>&1 | tee "$log"
done
