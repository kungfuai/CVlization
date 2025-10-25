# Training Pipeline Verification - Technical Reference

## Expected example.yaml Structure

```yaml
name: <example-name>                    # Lowercase with hyphens
capability: <category>/<subcategory>    # e.g., perception/object_detection
modalities:                             # Input/output types
  - vision
  - text
datasets:                               # Datasets used
  - <dataset-name>
stability: stable|experimental          # Maturity level
resources:                              # Hardware requirements
  gpu: 1
  vram_gb: 8-24
  disk_gb: 5-50
presets:                                # CVL command presets
  build:
    script: build.sh
    description: Build the Docker image
  train:
    script: train.sh
    description: Run training
tags:                                   # Searchable keywords
  - framework-name
  - model-architecture
tasks:                                  # ML tasks
  - task_name
frameworks:                             # ML frameworks used
  - pytorch
  - tensorflow
description: Brief description of what the example does
```

## Docker Mount Structure

Standard mount points in train.sh:

```bash
docker run --rm --gpus=all \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \           # Example directory (writable)
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \  # CVL package (read-only)
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \  # Model cache
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  "$IMG" python train.py "$@"
```

**Key points:**
- `/workspace` is the working directory (example folder, writable)
- `/cvlization_repo` contains the CVL Python package (read-only)
- HuggingFace cache is shared to avoid re-downloading models
- `PYTHONPATH` includes CVL package for imports

## Metric Ranges by Task

### Language Models (GPT, BERT, etc.)
- **Initial loss:** 8-12 (random initialization)
- **After 1 epoch:** 4-8
- **Well-trained:** 1-3
- **Overfitted:** <0.5
- **Perplexity:** exp(loss), lower is better

### Image Classification
- **Initial loss:** ~log(num_classes)
- **Initial accuracy:** ~1/num_classes (random)
- **Decent model:** >70% accuracy
- **Good model:** >85% accuracy
- **State-of-art:** >95% (ImageNet-like datasets)

### Object Detection (COCO-style)
- **mAP@50:95:** Main metric, 0-100 scale
  - <20: Poor detection
  - 20-40: Decent detection
  - 40-60: Good detection
  - >60: Strong detection
- **mAP@50:** More lenient (allows less precise boxes)
- **Loss components:** Total loss = classifier + bbox + objectness + RPN

### Segmentation
- **mIoU (mean Intersection over Union):** 0-1 or 0-100%
  - <0.3: Poor segmentation
  - 0.3-0.5: Decent segmentation
  - 0.5-0.7: Good segmentation
  - >0.7: Excellent segmentation
- **Pixel accuracy:** Simpler metric, typically higher than IoU
- **Dice coefficient:** Similar to IoU, 2*|X∩Y|/(|X|+|Y|)

### Fine-tuning Tasks
- **Train loss:** Should decrease steadily
- **Eval loss:** Should track train loss (within 0.1-0.5)
- **If eval >> train:** Overfitting, reduce training or add regularization
- **If both plateau:** Need more epochs or higher learning rate

## Log File Locations

Different frameworks use different logging conventions:

### PyTorch Lightning
```
lightning_logs/
└── version_0/
    ├── events.out.tfevents.*  # TensorBoard logs
    ├── hparams.yaml           # Hyperparameters
    └── metrics.csv            # Metrics in CSV format
```

### Weights & Biases
```
wandb/
└── run-<timestamp>-<id>/
    ├── files/
    │   ├── config.yaml        # Run configuration
    │   ├── wandb-metadata.json
    │   └── wandb-summary.json # Final metrics
    └── logs/
        └── debug.log
```

### Custom Training Scripts
```
logs/
├── train.log                   # Training logs
└── events.out.tfevents.*      # TensorBoard (if used)

outputs/
├── checkpoints/               # Model checkpoints
│   ├── epoch_1.pth
│   └── best_model.pth
└── lora_adapters/             # LoRA/PEFT adapters
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Common Environment Variables

### Training Control
```bash
BATCH_SIZE=2              # Batch size per GPU
GRAD_ACCUM=8              # Gradient accumulation steps
NUM_EPOCHS=10             # Number of training epochs
LR=1e-4                   # Learning rate
MAX_TRAIN_SAMPLES=100     # Limit training samples (for testing)
```

### Model Configuration
```bash
MODEL_ID=org/model-name   # HuggingFace model ID
MAX_SEQ_LEN=2048          # Maximum sequence length (LLMs)
LORA_R=16                 # LoRA rank
LORA_ALPHA=32             # LoRA alpha
USE_DORA=true             # Enable DoRA
```

### Data Configuration
```bash
TRAIN_DATA=path/or/hf_dataset  # Training data source
TRAIN_SPLIT=train              # Dataset split
VAL_SPLIT=validation           # Validation split
OUTPUT_DIR=outputs/run_name    # Output directory
```

### Framework Settings
```bash
PYTHONUNBUFFERED=1        # Disable Python output buffering
WANDB_API_KEY=<key>       # Weights & Biases API key
CUDA_VISIBLE_DEVICES=0    # GPU selection
```

## Docker Debugging Commands

### Inspect running container
```bash
# List containers
docker ps -a

# View logs
docker logs <container-id>

# Tail logs
docker logs -f <container-id>

# Execute command in container
docker exec -it <container-id> bash

# Check GPU visibility
docker exec <container-id> nvidia-smi

# Check mounts
docker inspect <container-id> | grep -A 20 Mounts
```

### Common issues
```bash
# Issue: Container exits immediately
docker logs <container-id>  # Check for errors

# Issue: Can't access files
docker exec -it <container-id> ls -la /workspace

# Issue: GPU not available
docker exec <container-id> nvidia-smi  # Should show GPU

# Issue: Import errors
docker exec <container-id> python -c "import cvlization"
```

## Performance Tuning

### GPU Memory Optimization
```bash
# If OOM (Out of Memory):
1. Reduce BATCH_SIZE (halve it)
2. Reduce MAX_SEQ_LEN (for LLMs)
3. Reduce image size (for vision models)
4. Increase GRAD_ACCUM to maintain effective batch size
5. Enable gradient checkpointing (if supported)
6. Use mixed precision training (fp16/bf16)
```

### Training Speed Optimization
```bash
# If training is slow:
1. Increase BATCH_SIZE (if GPU memory allows)
2. Enable mixed precision (bf16 > fp16 > fp32)
3. Reduce validation frequency
4. Use DataLoader num_workers > 0
5. Enable compile (PyTorch 2.0+)
6. Check nvidia-smi GPU utilization (should be >70%)
```

### Data Loading Optimization
```bash
# If GPU is idle between batches:
1. Increase DataLoader num_workers
2. Enable persistent_workers=True
3. Use pin_memory=True
4. Prefetch data to GPU
5. Check disk I/O (use SSD if possible)
```

## Verification Scripts

### Quick health check
```bash
#!/bin/bash
# Check if example is properly structured
EXAMPLE_DIR=$1
echo "Checking $EXAMPLE_DIR..."

required_files=("example.yaml" "Dockerfile" "build.sh" "train.sh" "train.py")
for file in "${required_files[@]}"; do
    if [ -f "$EXAMPLE_DIR/$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file MISSING"
    fi
done
```

### Monitor training metrics
```bash
#!/bin/bash
# Monitor key metrics during training
EXAMPLE_DIR=$1
LOG_FILE="$EXAMPLE_DIR/logs/train.log"

# Wait for log file
while [ ! -f "$LOG_FILE" ]; do sleep 1; done

# Monitor loss
tail -f "$LOG_FILE" | grep -i --line-buffered "loss\|accuracy\|map\|iou"
```

## Framework-Specific Notes

### PyTorch + Lightning
- Uses `lightning_logs/` for outputs
- Metrics in `metrics.csv` format
- Checkpoints in `checkpoints/` subdirectory
- Configure callbacks for custom behavior

### TRL (Transformers Reinforcement Learning)
- Used for LLM fine-tuning
- Supports LoRA/QLoRA/DoRA adapters
- Saves adapters separately from base model
- Integrates with HuggingFace Hub

### MMDetection / MMSegmentation
- Uses config-based system
- Checkpoints in `work_dirs/`
- Logs in JSON format
- Supports distributed training

### TorchVision Models
- Simple training loops
- Uses standard PyTorch optimizers
- Checkpoints saved manually
- Metrics logged to stdout or TensorBoard

## Additional Resources

- CVlization Documentation: `/docs` directory in repo
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- TRL Documentation: https://huggingface.co/docs/trl
- Weights & Biases: https://docs.wandb.ai/
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
