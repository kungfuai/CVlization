#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

set -e

echo "=== GPT-OSS GRPO Smoke Test ==="

# Backup original config
cp config.yaml config.yaml.bak

# Temporarily reduce max_steps for smoke test
sed 's/max_steps: 10/max_steps: 2/' config.yaml > config_test.yaml

# Run smoke test
echo "Running 2-step GRPO training for smoke test..."
docker run --runtime nvidia \
    --rm \
    -v $(pwd):/workspace \
    -v $REPO_ROOT/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    gpt_oss_grpo \
    bash -c "cp config_test.yaml config.yaml && python3 train.py && rm -rf test-output"

# Restore original config
mv config.yaml.bak config.yaml
rm config_test.yaml

echo "✅ Smoke test passed!"
