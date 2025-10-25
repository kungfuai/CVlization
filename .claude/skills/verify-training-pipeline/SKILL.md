---
name: verify-training-pipeline
description: Verify a CVlization training pipeline example is properly structured, can build, trains successfully, and logs appropriate metrics. Use when validating example implementations or debugging training issues.
---

# Verify Training Pipeline

Systematically verify that a CVlization training example is complete, properly structured, and functional.

## When to Use

- Validating a new or modified training example
- Debugging training pipeline issues
- Ensuring example completeness before commits
- Verifying example works after CVlization updates

## Verification Checklist

### 1. Structure Verification

Check that the example directory contains all required files:

```bash
# Navigate to example directory
cd examples/<capability>/<task>/<framework>/

# Expected structure:
# .
# ├── example.yaml        # Required: CVL metadata
# ├── Dockerfile          # Required: Container definition
# ├── build.sh            # Required: Build script
# ├── train.sh            # Required: Training script
# ├── train.py            # Required: Training code
# ├── README.md           # Recommended: Documentation
# ├── requirements.txt    # Optional: Python dependencies
# ├── data/               # Optional: Data directory
# └── outputs/            # Created at runtime
```

**Key files to check:**
- `example.yaml` - Must have: name, capability, stability, presets (build, train)
- `Dockerfile` - Should copy necessary files and install dependencies
- `build.sh` - Must set `SCRIPT_DIR` and call `docker build`
- `train.sh` - Must mount volumes correctly and pass environment variables

### 2. Build Verification

```bash
# Option 1: Build using script directly
./build.sh

# Option 2: Build using CVL CLI (recommended)
cvl run <example-name> build

# Verify image was created
docker images | grep <example-name>

# Expected: Image appears with recent timestamp
```

**What to check:**
- Build completes without errors (both methods)
- All dependencies install successfully
- Image size is reasonable (check for unnecessary files)
- `cvl info <example-name>` shows correct metadata

### 3. Training Verification

Start training and monitor for proper initialization:

```bash
# Option 1: Run training using script directly
./train.sh

# Option 2: Run training using CVL CLI (recommended)
cvl run <example-name> train

# With custom parameters (if supported)
BATCH_SIZE=2 NUM_EPOCHS=1 ./train.sh
```

**Immediate checks (first 30-60 seconds):**
- Container starts without errors
- Dataset loads successfully
- Model initializes (check GPU memory with `nvidia-smi`)
- Training loop begins (first batch processes)
- Logs are being written

### 4. Metrics Verification

Monitor metrics appropriate to the task type:

#### Generative Tasks (LLM, Text Generation, Image Generation)
- **Primary metric:** `train/loss` (should decrease over time)
- **Target:** Loss consistently decreasing, not NaN/Inf
- **Typical range:** Depends on task (LLM: 2-5 initial, <1 after convergence)
- **Check for:** Gradient explosions, NaN losses

```bash
# For LLM/generative models
tail -f logs/train.log | grep -i "loss\|iter\|step"
```

#### Classification Tasks (Image, Text, Document)
- **Primary metrics:** `train/loss`, `train/accuracy`, `val/accuracy`
- **Target:** Accuracy increasing, loss decreasing
- **Typical range:** Accuracy 0-100%, converges based on task difficulty
- **Check for:** Overfitting (train acc >> val acc)

```bash
# Watch accuracy metrics
tail -f lightning_logs/version_0/metrics.csv
# or for WandB
tail -f logs/train.log | grep -i "accuracy\|acc"
```

#### Object Detection Tasks
- **Primary metrics:** `train/loss`, `val/map` (mean Average Precision), `val/map_50`
- **Target:** mAP increasing, loss decreasing
- **Typical range:** mAP 0-100, good models achieve 30-90% depending on dataset
- **Components:** `loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`

```bash
# Monitor detection metrics
tail -f logs/train.log | grep -i "map\|loss_classifier\|loss_box"
```

#### Segmentation Tasks (Semantic, Instance, Panoptic)
- **Primary metrics:** `train/loss`, `val/iou` (Intersection over Union), `val/pixel_accuracy`
- **Target:** IoU increasing (>0.5 is decent, >0.7 is good), loss decreasing
- **Typical range:** IoU 0-1, pixel accuracy 0-100%
- **Variants:** mIoU (mean IoU across classes)

```bash
# Monitor segmentation metrics
tail -f lightning_logs/version_0/metrics.csv | grep -i "iou\|pixel"
```

#### Fine-tuning / Transfer Learning
- **Primary metrics:** `train/loss`, `eval/loss`, task-specific metrics
- **Target:** Both losses decreasing, eval loss not diverging from train loss
- **Check for:** Catastrophic forgetting, adapter convergence
- **Special:** For LoRA/DoRA, verify adapters are saved

```bash
# Check if adapters are being saved
ls -la outputs/*/lora_adapters/
# Should contain: adapter_config.json, adapter_model.safetensors
```

### 5. Runtime Checks

**GPU Utilization:**
```bash
# In another terminal
watch -n 1 nvidia-smi

# Expected:
# - GPU memory usage 60-95% (adjust batch size if 100% or <30%)
# - GPU utilization 70-100%
# - Temperature stable (<85°C)
```

**Docker Container Health:**
```bash
# List running containers
docker ps

# Check logs for errors
docker logs <container-name-or-id>

# Verify mounts
docker inspect <container-id> | grep -A 10 Mounts
# Should see: workspace, cvlization_repo, huggingface cache
```

**Output Directory:**
```bash
# Check outputs are being written
ls -la outputs/ logs/ lightning_logs/
# Expected: Checkpoints, logs, or saved models appearing

# For WandB integration
ls -la wandb/
# Expected: run-<timestamp>-<id> directories
```

### 6. Quick Validation Test

For fast verification (useful during development):

```bash
# Run 1 epoch with limited data
MAX_TRAIN_SAMPLES=10 NUM_EPOCHS=1 ./train.sh

# Expected runtime: 1-5 minutes
# Verify: Completes without errors, metrics logged
```

## Common Issues and Fixes

### Build Failures
```bash
# Issue: Dockerfile can't find files
# Fix: Check COPY paths are relative to Dockerfile location

# Issue: Dependency conflicts
# Fix: Check requirements.txt versions, update base image

# Issue: Large build context
# Fix: Add .dockerignore file
```

### Training Failures
```bash
# Issue: CUDA out of memory
# Fix: Reduce BATCH_SIZE, MAX_SEQ_LEN, or image size

# Issue: Dataset not found
# Fix: Check data/ directory exists, run data preparation script

# Issue: Permission denied on outputs
# Fix: Ensure output directories are created before docker run
```

### Metric Issues
```bash
# Issue: Loss is NaN
# Fix: Reduce learning rate, check data normalization, verify labels

# Issue: No metrics logged
# Fix: Check training script has logging configured (wandb/tensorboard)

# Issue: Loss not decreasing
# Fix: Verify learning rate, check data quality, increase epochs
```

## Example Commands

### Perception - Object Detection
```bash
cd examples/perception/object_detection/torchvision
./build.sh
./train.sh
# Monitor: train/loss, val/map, val/map_50
# Success: mAP > 0.3 after a few epochs
```

### Perception - Semantic Segmentation
```bash
cd examples/perception/segmentation/semantic_torchvision
./build.sh
./train.sh
# Monitor: train/loss, val/iou, val/pixel_accuracy
# Success: IoU > 0.5, pixel_accuracy > 80%
```

### Generative - LLM Training
```bash
cd examples/generative/llm/nanogpt
./build.sh
./train.sh
# Monitor: train/loss, val/loss, iter time
# Success: Loss decreasing from ~4.0 to <2.0
```

### Document AI - Fine-tuning
```bash
cd examples/perception/doc_ai/granite_docling_finetune
./build.sh
MAX_TRAIN_SAMPLES=20 NUM_EPOCHS=1 ./train.sh
# Monitor: train/loss, eval/loss
# Success: Both losses decrease, adapters saved to outputs/
```

## CVL Integration

These examples integrate with the CVL command system:

```bash
# List all available examples
cvl list

# Get example info
cvl info granite_docling_finetune

# Run example directly (uses example.yaml presets)
cvl run granite_docling_finetune build
cvl run granite_docling_finetune train
```

## Success Criteria

A training pipeline passes verification when:

1. ✅ **Structure**: All required files present, example.yaml valid
2. ✅ **Build**: Docker image builds without errors (both `./build.sh` and `cvl run <name> build`)
3. ✅ **Start**: Training starts, dataset loads, model initializes (both `./train.sh` and `cvl run <name> train`)
4. ✅ **Metrics Improve**: Training loss decreases OR model accuracy/mAP/IoU improves over epochs
5. ✅ **Outputs**: Checkpoints/adapters/logs saved to outputs/
6. ✅ **CVL CLI**: `cvl info <name>` shows correct metadata, build and train presets work
7. ✅ **Documentation**: README explains how to use the example

## Related Files

Check these files for debugging:
- `train.py` - Core training logic
- `Dockerfile` - Environment setup
- `requirements.txt` - Python dependencies
- `example.yaml` - CVL metadata and presets
- `README.md` - Usage instructions

## Tips

- Use `MAX_TRAIN_SAMPLES=<small_number>` for fast validation
- Monitor GPU memory with `nvidia-smi` in separate terminal
- Check `docker logs <container>` if training hangs
- For WandB integration, set `WANDB_API_KEY` environment variable
- Most examples support environment variable overrides (check train.sh)
