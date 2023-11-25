Adapted from: https://github.com/chenxwh/cog-deforum-stable-diffusion/tree/replicate

## Quickstart

[Install `cog`](https://github.com/replicate/cog).


Download model weights:

```
python scripts/download-weights
```

Then,

```
cog predict
```

<!-- ```
pip install git+https://github.com/openai/CLIP.git
pip install jsonmerge clean-fid resize-right torchdiffeq torchsde pydantic omegaconf
pip install open-clip-torch numexpr
``` -->

## Running on a GPU with 12GB VRAM

Currently stuck at the ffmpeg step.