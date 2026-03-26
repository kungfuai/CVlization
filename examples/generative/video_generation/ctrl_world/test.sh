#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Ctrl-World smoke test..."
echo "  (Replays 1 trajectory with 3 interaction steps for quick validation)"

OUTPUT_DIR="${SCRIPT_DIR}/outputs/ctrl_world_outputs"

# Clean previous test outputs
rm -rf "$OUTPUT_DIR"

# Run replay on a single trajectory with minimal steps
"${SCRIPT_DIR}/predict.sh" \
  --interact_num 3 \
  --output ctrl_world_outputs

# Verify output
MP4_FILES=$(find "$OUTPUT_DIR" -name "replay_899_*.mp4" -type f 2>/dev/null)
if [[ -n "$MP4_FILES" ]]; then
  echo "SUCCESS: Generated replay video(s):"
  echo "$MP4_FILES" | xargs ls -lh
else
  echo "FAILED: No output video found in $OUTPUT_DIR"
  exit 1
fi

echo "Smoke test passed!"
