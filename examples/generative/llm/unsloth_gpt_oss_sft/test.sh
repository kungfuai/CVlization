#!/bin/bash
# Smoke test for gpt_oss_finetune
# Validates config loading and runs 2 training steps

set -e

echo "=== GPT-OSS 20B Fine-tuning Smoke Test ==="
echo "This will run 2 training steps with 100 samples to verify the setup works."
echo

# Create test config
cat > config_test.yaml <<EOF
dataset:
  path: "HuggingFaceH4/Multilingual-Thinking"
  format: "sharegpt"
  split: "train"
  max_samples: 100

model:
  name: "unsloth/gpt-oss-20b"
  max_seq_length: 512
  load_in_4bit: true

lora:
  r: 8
  alpha: 8
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
    -v $(pwd)/../../../data/container_cache:/root/.cache \
    -e HF_TOKEN=$HF_TOKEN \
    gpt_oss_finetune \
    bash -c "cp config_test.yaml config.yaml && python3 train.py && rm -rf test-output"

# Restore original config and cleanup
mv config.yaml.bak config.yaml
rm config_test.yaml

echo
echo "✅ Smoke test passed!"
