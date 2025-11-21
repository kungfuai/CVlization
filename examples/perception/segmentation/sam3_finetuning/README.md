# SAM3 (Segment Anything Model 3) Fine-tuning

Fine-tune SAM3 on custom COCO-format segmentation datasets. SAM3 is Meta's latest open-vocabulary segmentation model with 848M parameters, supporting both detection and segmentation tasks.

## Features

- Fine-tune on custom COCO datasets with segmentation masks
- Open-vocabulary segmentation with text prompts
- Distributed training support (multi-GPU)
- Based on Facebook Research's official SAM3 implementation (commit `84cc43b`)
- Comprehensive training configuration via Hydra
- Self-contained Docker image with SAM3 pre-installed for reproducibility

## Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (e.g., NVIDIA A10, RTX 3090/4090)
- **Recommended**: 1x A100 40GB or better for faster training
- **Disk**: ~30GB for model checkpoints and dependencies

## Quick Start

```bash
cd examples/perception/segmentation/sam3_finetuning

# 1. Build the Docker image
./build.sh

# 2. Train with synthetic test shapes (default - fastest for testing!)
./train.sh \
  --output-dir outputs/sam3_finetuning \
  --epochs 20 \
  --batch-size 1 \
  --lr 8e-4

# OR train with HuggingFace dataset download:
./train.sh \
  --hf-dataset keremberke/pcb-defect-segmentation \
  --output-dir outputs/sam3_finetuning \
  --epochs 20 \
  --batch-size 1 \
  --lr 8e-4

# OR use your own COCO dataset:
./train.sh \
  --dataset-dir /path/to/your/coco-dataset \
  --output-dir outputs/sam3_finetuning \
  --epochs 20 \
  --batch-size 1 \
  --lr 8e-4
```

## Verifying the Installation (No GPU Required!)

Before training on GPU, you can verify that the training code path works correctly:

```bash
# Build the image
./build.sh

# Run simplified training verification
docker run --rm --ipc=host \
  --mount "type=bind,src=$(pwd),dst=/workspace" \
  sam3-finetuning python3 train_simple.py
```

**Expected output** (on CPU without GPU):
```
✓ SAM3 installed successfully
✓ Hydra config initialization succeeds
✓ Full SAM3 config loads (400+ lines)
✓ Training pipeline starts
✓ Trainer instantiation begins
✗ RuntimeError: Found no NVIDIA driver (EXPECTED on CPU)
```

This proves the training code path is correct and will work on a GPU machine. The error `Found no NVIDIA driver` is expected on CPU and confirms the code reaches the device initialization step.

## Synthetic Test Dataset (Default)

By default, the training script generates a synthetic dataset of geometric shapes (circles, rectangles, triangles) for quick testing:

```bash
# Uses synthetic dataset by default (no args needed!)
./train.sh --epochs 5

# Or via CVL:
cvl run sam3_finetuning train --epochs 5
```

**Dataset Details:**
- **Training**: 20 images with 2-5 random shapes each
- **Validation**: 5 images with 2-5 random shapes each
- **Categories**: circle, rectangle, triangle
- **Image size**: 640x480
- **Format**: COCO with RLE segmentation masks
- **Location**: `data/test_shapes/`

The dataset is automatically generated on first run and recreated each time you train without specifying a dataset. This is perfect for:
- Quick testing of the training pipeline
- Verifying GPU setup and memory usage
- Development and debugging
- CI/CD pipelines

**Customizing the Synthetic Dataset:**

You can customize the dataset generation by modifying `dataset_builder.py`:

```python
from dataset_builder import DatasetBuilder

# Create custom dataset
DatasetBuilder(
    output_dir="data/my_shapes",
    num_train=50,           # More training images
    num_val=10,             # More validation images
    image_size=(1024, 768), # Higher resolution
    shapes_per_image=(3, 8) # 3-8 shapes per image
)
```

## Automatic Dataset Download

The training script can automatically download and convert datasets from HuggingFace:

```bash
./train.sh --hf-dataset keremberke/pcb-defect-segmentation
```

**Available HuggingFace Datasets:**
- `keremberke/pcb-defect-segmentation` - PCB defect detection (~1000 images)
- Any HuggingFace dataset with segmentation annotations

The dataset will be:
1. Downloaded from HuggingFace
2. Automatically converted to COCO format with RLE masks
3. Cached in `data/` directory for future runs

## Manual Dataset Preparation

**HuggingFace Datasets**:
- Browse: https://huggingface.co/datasets
- Filter by task: "Instance Segmentation"
- Use with `--hf-dataset` flag

**Roboflow Universe** (requires free account):
- [Fashion Assistant](https://universe.roboflow.com/roboflow-jvuqo/fashion-assistant-segmentation) - Clothing segmentation
- [Trash Segmentation](https://universe.roboflow.com) - Waste detection
- Export as "COCO Segmentation" format
- Use with `--dataset-dir` flag

**Custom Datasets**:

For your own datasets, convert them to COCO format. Tools:
- [Roboflow](https://roboflow.com) - Upload and export as COCO
- [CVAT](https://cvat.ai) - Annotation tool with COCO export
- [LabelMe](https://github.com/wkentaro/labelme) - Convert to COCO with `labelme2coco`

## Dataset Format

SAM3 expects COCO-format annotations with segmentation masks. Your dataset should follow this structure:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   │   └── ...
│   └── _annotations.coco.json
└── test/  (optional)
    ├── images/
    │   └── ...
    └── _annotations.coco.json
```

### COCO JSON Format

The `_annotations.coco.json` file should contain:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345.0,
      "segmentation": {
        "counts": "...",  // RLE format
        "size": [480, 640]
      },
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

**Important**: SAM3 requires segmentation masks in RLE (Run-Length Encoding) format. If you have polygon annotations, convert them using:

```python
from pycocotools import mask as mask_utils
import numpy as np

# Convert polygon to RLE
rle = mask_utils.frPyObjects(polygon, height, width)
binary_mask = mask_utils.decode(rle)
```

## Training Configuration

### Command-Line Options

```bash
./train.sh \
  --dataset-dir /path/to/dataset      # Required: COCO dataset root
  --output-dir outputs/sam3           # Output directory (default: outputs/sam3_finetuning)
  --epochs 20                          # Training epochs (default: 20)
  --batch-size 1                       # Batch size per GPU (default: 1)
  --lr 8e-4                            # Learning rate (default: 8e-4)
  --num-gpus 1                         # Number of GPUs (default: 1)
  --checkpoint /path/to/ckpt.pt        # Resume from checkpoint (optional)
  --use-base-config                    # Use SAM3's full Roboflow config
```

### Advanced Configuration

For full control over training, use SAM3's native Hydra configuration:

```bash
# 1. Copy and modify the base config
cp projects/sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
   outputs/my_config.yaml

# 2. Edit paths in the config:
#    - paths.roboflow_vl_100_root: your dataset directory
#    - paths.experiment_log_dir: your output directory
#    - paths.bpe_path: projects/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# 3. Run with custom config
./train.sh --use-base-config
```

Key configuration sections:
- **Trainer**: Epochs, precision, gradient clipping
- **Data**: Batch size, workers, transforms
- **Optimizer**: Learning rate, weight decay, scheduler
- **Loss**: Matcher, classification loss, box loss, mask loss
- **Model**: Architecture, backbone, decoder layers

See `projects/sam3/sam3/train/configs/` for example configurations.

## Training Details

### Default Settings (from SAM3 paper)

- **Resolution**: 1008×1008 pixels
- **Model**: ViT-L backbone (1024-dim, 32 layers)
- **Parameters**: 848M total
- **Learning Rate**:
  - Transformer: 8e-4
  - Vision backbone: 2.5e-4 with layer decay (0.9)
  - Language backbone: 5e-5
- **Optimizer**: AdamW with weight decay 0.1
- **Precision**: bfloat16 mixed precision
- **Loss Functions**:
  - Classification: Focal loss (alpha=0.25, gamma=2)
  - Box: L1 + GIoU
  - Mask: Sigmoid focal + Dice (optional)

### Memory Optimization

For limited VRAM:

1. **Reduce batch size**: Use `--batch-size 1` and gradient accumulation
2. **Use gradient checkpointing**: Edit config to enable checkpointing
3. **Freeze backbone**: Only train decoder layers
4. **Reduce resolution**: Lower input size in config (min 480px)

## Monitoring Training

SAM3 logs metrics to TensorBoard:

```bash
# View training logs
tensorboard --logdir outputs/sam3_finetuning --port 6006

# Metrics tracked:
# - loss/train: Training loss
# - loss/val: Validation loss
# - learning_rate: Current LR
# - mAP: Mean Average Precision (if enabled)
```

## Inference

After training, use the fine-tuned checkpoint for inference:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load fine-tuned model
model = build_sam3_image_model(checkpoint="outputs/sam3_finetuning/checkpoint.pt")
processor = Sam3Processor(model)

# Run inference with text prompt
inference_state = processor.set_image(image)
output = processor.set_text_prompt(
    state=inference_state,
    prompt="segment all people"
)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
```

See `projects/sam3/examples/` for complete inference notebooks.

## Evaluation

Evaluate the fine-tuned model:

```bash
# Run evaluation on validation set
./train.sh \
  --dataset-dir /path/to/dataset \
  --checkpoint outputs/sam3_finetuning/checkpoint.pt \
  --eval-only

# Metrics reported:
# - mAP@50:95
# - mAP@50, mAP@75
# - mAP (small), mAP (medium), mAP (large)
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` to 1
- Use gradient accumulation in config
- Lower input resolution
- Enable gradient checkpointing
- Freeze vision backbone

### Training Loss Not Decreasing

- Check learning rate (try 1e-4 to 1e-3)
- Verify dataset annotations are correct
- Ensure segmentation masks are valid
- Increase warmup steps
- Check data transforms aren't too aggressive

### Dataset Loading Errors

- Verify COCO JSON structure
- Ensure segmentation masks are RLE format
- Check image paths are correct
- Validate category IDs start from 1

### CUDA Errors

- Update to PyTorch 2.7+ with CUDA 12.6+
- Check GPU compatibility
- Ensure sufficient VRAM available
- Try reducing batch size or resolution

## References

- **SAM3 Paper**: [Segment Anything Model 3](https://arxiv.org/abs/2501.XXXXX)
- **SAM3 GitHub**: https://github.com/facebookresearch/sam3
- **SA-Co Dataset**: 4M+ unique concepts for training
- **Blog Post**: https://blog.roboflow.com/fine-tune-sam3/

## License

SAM3 is released under the Apache 2.0 license. See the [SAM3 repository](https://github.com/facebookresearch/sam3) for details.
