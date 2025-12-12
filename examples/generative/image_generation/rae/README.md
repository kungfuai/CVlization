# RAE: Representation Autoencoders for Diffusion Transformers

Generate high-quality images using Representation Autoencoders (RAE) with Diffusion Transformers (DiT).

## Overview

RAE uses pretrained visual encoders (DINOv2, SigLIP2, etc.) as frozen encoders paired with trained ViT decoders. A Stage 2 diffusion transformer generates images in the learned latent space.

Key results from the paper:
- State-of-the-art FID on ImageNet-256
- Leverages pretrained representations for efficient training
- Two-stage pipeline: RAE encoder/decoder + DiT diffusion model

## Quick Start

### Build

```bash
cvl run rae build
```

### Generate Images

Generate images using pretrained DiT-XL model (auto-downloads from HuggingFace):

```bash
cvl run rae generate

# Generate specific ImageNet classes
cvl run rae generate -- --class-ids 207,388,971,985 --num-samples 8

# More sampling steps for higher quality
cvl run rae generate -- --num-steps 100 --cfg-scale 2.0
```

### Smoke Test

Run a quick generation test:

```bash
cvl run rae test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAE Generation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Noise z ~ N(0,1)                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   DiT-XL/DDT    │  <-- Stage 2: Diffusion in latent space   │
│  │   (pretrained)  │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  Latent z' (768 x 16 x 16)                                      │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   RAE Decoder   │  <-- Stage 1: Latent → Image              │
│  │   (ViT-XL)      │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  Image (256 x 256 x 3)                                          │
│                                                                 │
│  The RAE encoder (DINOv2) is only used during training          │
│  to compute the latent space; inference only needs the decoder  │
└─────────────────────────────────────────────────────────────────┘
```

## Generation Arguments

```bash
cvl run rae generate -- \
    --num-samples 8 \
    --class-ids 207,388 \
    --cfg-scale 1.8 \
    --num-steps 50 \
    --mode ode \
    --seed 42
```

Key arguments:
- `--num-samples`: Number of images to generate (default: 8)
- `--class-ids`: Comma-separated ImageNet class IDs
- `--cfg-scale`: Classifier-free guidance scale (default: 1.8)
- `--num-steps`: Sampling steps (default: 50)
- `--mode`: ode or sde sampling (default: ode)
- `--seed`: Random seed for reproducibility

## Pretrained Models

Models are automatically downloaded from [HuggingFace](https://huggingface.co/nyu-visionx/RAE-collections):

| Component | Description | Size |
|-----------|-------------|------|
| RAE Decoder | ViT-XL decoder for DINOv2-B | ~1.2GB |
| DiT-XL | Diffusion transformer | ~2.5GB |
| Stats | Latent normalization statistics | ~1MB |

## Resources

- **VRAM**: ~16-20GB for inference
- **Disk**: ~5GB for pretrained models
- **Generation time**: ~10-30 seconds per image (depending on steps)

## ImageNet Class IDs

Some example class IDs to try:
- 207: Golden retriever
- 360: Otter
- 387: Lesser panda
- 388: Giant panda
- 417: Balloon
- 971: Bubble
- 972: Cliff
- 979: Valley
- 985: Daisy

## References

- Paper: [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)
- Code: [github.com/bytetriper/RAE](https://github.com/bytetriper/RAE)
- Authors: Boyang Zheng, Nanye Ma, Shengbang Tong, Saining Xie (NYU)

## Citation

```bibtex
@article{zheng2024rae,
  title={Diffusion Transformers with Representation Autoencoders},
  author={Zheng, Boyang and Ma, Nanye and Tong, Shengbang and Xie, Saining},
  journal={arXiv preprint arXiv:2510.11690},
  year={2024}
}
```
