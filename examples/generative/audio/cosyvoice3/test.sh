#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running CosyVoice3 smoke test..."

# Use outputs directory within workspace (mounted in container)
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/test_output.wav"
# Clean up on exit (may fail if file is root-owned from container)
trap "rm -f $OUTPUT_FILE 2>/dev/null || true" EXIT

# Run inference with default sample (use relative path for container)
"${SCRIPT_DIR}/predict.sh" \
  --text "Hello, this is a test of the CosyVoice text to speech system." \
  --output "outputs/test_output.wav" \
  --mode zero_shot

# Verify output exists and has non-zero size
if [[ -f "$OUTPUT_FILE" ]] && [[ -s "$OUTPUT_FILE" ]]; then
  echo "SUCCESS: Audio file generated"
  ls -la "$OUTPUT_FILE"
else
  echo "FAILED: Output file not found or empty"
  exit 1
fi

echo "Smoke test passed!"
