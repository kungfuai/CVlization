#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "ZAI_API_KEY is required. Set it before running tests."
  echo "Get an API key at https://open.z.ai"
  exit 1
fi

echo "Running GLM-5V-Turbo smoke test..."

bash "$SCRIPT_DIR/predict.sh" \
  --image test_images/sample.jpg \
  --task caption \
  --max-tokens 256 \
  --output "outputs/test_caption.txt"

echo "Smoke test completed successfully!"
