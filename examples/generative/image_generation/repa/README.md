# REPA: Representation Alignment for Generation

Training diffusion transformers with representation alignment for faster convergence and better image quality.

## Overview

REPA (REPresentation Alignment) aligns noisy input states in diffusion models with representations from pretrained visual encoders (DINOv2, CLIP, etc.). This significantly improves training efficiency and generation quality.

Key results from the paper:
- 17.5x faster training compared to standard SiT
- State-of-the-art FID=1.42 on ImageNet-256

## Quick Start

### Build

```bash
cvl run repa build
```

### Generate Images (Pretrained Model)

Generate images using the pretrained SiT-XL/2 model (auto-downloads):

```bash
cvl run repa generate

# Generate specific ImageNet classes
cvl run repa generate -- --class-ids 207,388,971,985 --num-samples 4

# More sampling steps for higher quality
cvl run repa generate -- --num-steps 500 --cfg-scale 2.0
```

### Train on CIFAR-10

Train a smaller model on CIFAR-10 (auto-downloads ~170MB):

```bash
cvl run repa train

# Custom training parameters
cvl run repa train -- --model SiT-B/2 --max-train-steps 50000 --batch-size 64
```

### Smoke Test

Run a quick training test (500 steps):

```bash
cvl run repa test
```

## Training

### Datasets

| Dataset | Classes | Resolution | Download |
|---------|---------|------------|----------|
| CIFAR-10 | 10 | 32→256 | Auto (~170MB) |
| CIFAR-100 | 100 | 32→256 | Auto (~170MB) |
| ImageNet | 1000 | 256 | HuggingFace (gated, ~150GB) |

For CIFAR-10/100, images are resized to 256x256 and VAE latents are computed on-the-fly.

### ImageNet Setup (Gated Dataset)

ImageNet is loaded from [HuggingFace](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and requires:

1. **Create HuggingFace account** at https://huggingface.co/join
2. **Accept ImageNet terms** at https://huggingface.co/datasets/ILSVRC/imagenet-1k (click "Access repository")
3. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```
4. **Run training**:
   ```bash
   cvl run repa train -- --dataset imagenet --model SiT-XL/2
   ```

Note: ImageNet download is ~150GB and requires significant disk space.

### Models

| Model | Parameters | VRAM (fp16) |
|-------|------------|-------------|
| SiT-S/2 | ~33M | ~8GB |
| SiT-B/2 | ~130M | ~12GB |
| SiT-L/2 | ~458M | ~18GB |
| SiT-XL/2 | ~675M | ~24GB |

### Training Arguments

```bash
cvl run repa train -- \
    --dataset cifar10 \
    --model SiT-B/2 \
    --enc-type dinov2-vit-b \
    --proj-coeff 0.5 \
    --max-train-steps 100000 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --report-to wandb
```

Key arguments:
- `--dataset`: cifar10, cifar100, or imagenet
- `--model`: SiT-{S,B,L,XL}/2
- `--enc-type`: dinov2-vit-{s,b,l,g} (for REPA alignment)
- `--proj-coeff`: Weight for projection loss (default: 0.5)
- `--max-train-steps`: Number of training steps

## Generation

### Arguments

```bash
cvl run repa generate -- \
    --num-samples 8 \
    --class-ids 207,388 \
    --cfg-scale 1.8 \
    --num-steps 250 \
    --mode sde
```

Key arguments:
- `--num-samples`: Number of images to generate
- `--class-ids`: Comma-separated ImageNet class IDs
- `--cfg-scale`: Classifier-free guidance scale (default: 1.8)
- `--num-steps`: Sampling steps (default: 250)
- `--mode`: ode or sde sampling

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REPA Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image (256x256)                                                │
│       │                                                         │
│       ├──────────────────┐                                      │
│       │                  │                                      │
│       ▼                  ▼                                      │
│  ┌─────────┐       ┌──────────┐                                 │
│  │ SD-VAE  │       │  DINOv2  │  <-- PRETRAINED, FROZEN         │
│  │(frozen) │       │ (frozen) │                                 │
│  └────┬────┘       └────┬─────┘                                 │
│       │                 │                                       │
│       ▼                 │                                       │
│  Latent z (32x32x4)     │                                       │
│       │                 │                                       │
│       ▼                 │                                       │
│  ┌─────────────┐        │                                       │
│  │    SiT      │        │                                       │
│  │ (TRAINING)  │<───────┴── REPA Alignment Loss                 │
│  │ from scratch│                                                │
│  └─────────────┘                                                │
│                                                                 │
│  Loss = Denoising Loss + 0.5 * Projection Loss                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The REPA method's key insight: aligning the SiT's intermediate representations
with DINOv2 features provides a learning signal that accelerates training by 17.5x.

## Resources

- **VRAM**: 12-24GB depending on model size
- **Disk**: ~5GB for CIFAR-10, ~150GB for ImageNet
- **Training time**: ~2 hours for CIFAR-10 (50k steps), ~days for ImageNet

## References

- Paper: [Representation Alignment for Generation](https://arxiv.org/abs/2410.06940)
- Code: [github.com/sihyun-yu/REPA](https://github.com/sihyun-yu/REPA)
- Project page: [sihyun.me/REPA](https://sihyun.me/REPA)

## Citation

```bibtex
@inproceedings{yu2025repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2025},
  booktitle={International Conference on Learning Representations},
}
```
