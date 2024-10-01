## Quickstart

Install common dependencies like `torch`, `transformers`, `torchvision`, `einops`, `diffusers`, etc.

Then run one of the training scripts from the repository root:

```bash
python -m examples.image_gen.nano_diffusion.train4
```

## Methodology

By making incremental changes to create training pipeline variations, we can identify the necessary and sufficient factors for generation quality. The refactored pipelines can then be made minimalistic and scalable, while maintaining quality.

## References

- https://github.com/VSehwag/minimal-diffusion
- https://github.com/cloneofsimo/minDiffusion
- https://github.com/facebookresearch/DiT/blob/main/models.py