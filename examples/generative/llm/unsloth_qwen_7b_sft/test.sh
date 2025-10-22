#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Smoke test for qwen_7b_finetune
# Validates config loading and runs 2 training steps

set -e

echo "=== Qwen 2.5 7B Fine-tuning Smoke Test ==="
echo "This will run 2 training steps with 100 samples to verify the setup works."
echo

# Create test config
cat > config_test.yaml <<EOF
dataset:
  path: "yahma/alpaca-cleaned"
  format: "alpaca"
  split: "train"
  max_samples: 100

model:
  name: "unsloth/Qwen2.5-7B-Instruct"
  max_seq_length: 512
  load_in_4bit: true

lora:
  r: 16
  alpha: 16
  dropout: 0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  output_dir: "./test-output"
  max_steps: 2
  num_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-4
  warmup_steps: 0
  lr_scheduler_type: "linear"
  optim: "adamw_8bit"
  weight_decay: 0.01
  logging_steps: 1
  save_steps: 10
  eval_steps: 10
  do_eval: false
  seed: 3407
EOF

# Backup original config
cp config.yaml config.yaml.bak

# Run test
echo "Running test..."
docker run --runtime nvidia \
    -v $(pwd):/workspace \
    -v $REPO_ROOT/data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    qwen_7b_finetune \
    bash -c "cp config_test.yaml config.yaml && python3 train.py && rm -rf test-output"

# Restore original config and cleanup
mv config.yaml.bak config.yaml
rm config_test.yaml

echo
echo "✅ Smoke test passed!"
