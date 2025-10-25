# Segment Anything Model (SAM) Instance Segmentation

Fine-tune Meta's Segment Anything Model (SAM) using MobileSAM for instance segmentation on custom datasets.

## Overview

This example demonstrates how to fine-tune the Segment Anything Model (SAM) for instance segmentation tasks. It uses the lightweight MobileSAM variant (vit_t) which is optimized for efficient training and inference while maintaining strong segmentation performance.

**Key Features:**
- MobileSAM (Tiny ViT) architecture for efficient training
- Mixed precision training support
- Dice loss for segmentation optimization
- Automatic model weight and dataset downloading with caching
- Checkpoint saving (best and latest models)

## Quick Start

### Build the Docker Image

Using build script:
```bash
./build.sh
```

Or using CVL CLI:
```bash
cvl run sam build
```

### Run Training

Basic training with default settings:
```bash
./train.sh
```

Or using CVL CLI:
```bash
cvl run sam train
```

### Debug Mode (Fast Verification)

Train on a single example for 5 iterations to verify the setup:
```bash
./train.sh -d -i 5
```

## Configuration

### Command-line Arguments

The training script supports the following arguments:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--debug_with_one_example` | `-d` | False | Debug mode: train on single example |
| `--checkpoint_name` | `-c` | `sam_instance_seg` | Name for saved checkpoints |
| `--model_type` | `-m` | `vit_t` | SAM model variant (vit_t, vit_b, vit_h) |
| `--batch_size` | `-b` | 1 | Training batch size |
| `--n_objects_per_batch` | `-n` | 25 | Number of objects sampled per batch |
| `--device` | `-g` | `cuda` | Device for training (cuda/cpu) |
| `--n_iterations` | `-i` | 200 | Total training iterations |
| `--n_sub_iteration` | `-s` | 2 | Sub-iterations per main iteration |

### Example Configurations

**Fast training (testing):**
```bash
./train.sh -i 50 -n 10
```

**Extended training:**
```bash
./train.sh -i 1000 -n 50
```

**Using different SAM model:**
```bash
./train.sh -m vit_b -i 500
```

## Dataset

By default, this example uses the **Penn-Fudan Pedestrian Dataset** which includes:
- 170 images with pedestrian annotations
- Instance segmentation masks
- Suitable for person detection and segmentation tasks

The dataset is automatically downloaded on first run and cached for subsequent runs.

### Using Custom Datasets

To use your own dataset, modify `train.py` to use a different dataset builder:

```python
from cvlization.dataset.your_dataset import YourDatasetBuilder

dataset_builder = YourDatasetBuilder(
    flavor="torchvision",
    include_masks=True,
    label_offset=1,
    normalize_with_min_max=False,
)
```

## Training Metrics

The training logs the following metrics per epoch:
- **Dice Score**: Measures overlap between predicted and ground truth masks (lower is better in this implementation)
- **Training time per iteration**
- **Best/current metric tracking**

Example output:
```
Epoch 0: Dice Score = 0.909
Epoch 1: Dice Score = 0.894 (best: 0.091)
Epoch 2: Dice Score = 0.906
```

## Output

### Checkpoints

Trained models are saved to `checkpoints/sam_instance_seg/`:
- `best.pt` - Best model based on validation Dice score
- `latest.pt` - Most recent model checkpoint

### Model Weights Cache

Pre-trained SAM weights are cached to `~/.sam_models/`:
- `vit_t_mobile_sam.pth` (40.7 MB) - Downloaded on first run

## Architecture Details

**MobileSAM (Tiny ViT):**
- Lightweight variant of SAM optimized for edge deployment
- ~10x faster than SAM-ViT-B
- Maintains competitive accuracy for most segmentation tasks
- Image encoder frozen during fine-tuning (only mask decoder trained)

**Training Configuration:**
- Optimizer: Adam (lr=1e-5)
- Scheduler: ReduceLROnPlateau (factor=0.9, patience=10)
- Loss: Dice Loss (measures mask overlap)
- Mixed Precision: Enabled by default

## Resources

- **GPU**: 1 GPU required
- **VRAM**: ~16 GB recommended
- **Disk**: ~10 GB (model weights + dataset)
- **Docker Image**: ~8.2 GB

## Troubleshooting

**CUDA Out of Memory:**
```bash
# Reduce objects per batch
./train.sh -n 10

# Or reduce batch size (already at minimum)
./train.sh -b 1 -n 10
```

**Slow Downloads:**
- Model weights and datasets are cached after first download
- Check `~/.sam_models/` and `./data/` directories

**Permission Issues:**
```bash
chmod +x build.sh train.sh
```

## References

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI Research
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - Efficient SAM variant
- Penn-Fudan Pedestrian Database - University of Pennsylvania

## Related Examples

- `instance_torchvision` - Instance segmentation with Torchvision
- `instance_mmdet` - Instance segmentation with MMDetection
- `semantic_torchvision` - Semantic segmentation with Torchvision
