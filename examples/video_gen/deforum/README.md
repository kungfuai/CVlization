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

Tested on 3090 (12GB VRAM).

Or, initialize from an image:

```
cog predict -i use_init=true -i init_image=@<path_to_image>
```

<!-- ```
pip install git+https://github.com/openai/CLIP.git
pip install jsonmerge clean-fid resize-right torchdiffeq torchsde pydantic omegaconf
pip install open-clip-torch numexpr
``` -->
