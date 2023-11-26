Adapted from: https://github.com/chenxwh/cog-deforum-stable-diffusion/tree/replicate

## Generate a video using Deforum

This is tested on a GPU with 12GB VRAM.

[Install `cog`](https://github.com/replicate/cog).

Go to this directory:

```
cd examples/video_gen/deforum
```

Download model weights:

```
python scripts/download-weights
```

Then,

```
cog predict
```



Or, initialize from an image:

```
cog predict -i use_init=true -i init_image=@<path_to_image>
```

<!-- ```
pip install git+https://github.com/openai/CLIP.git
pip install jsonmerge clean-fid resize-right torchdiffeq torchsde pydantic omegaconf
pip install open-clip-torch numexpr
``` -->

Expect the output video `output.mp4` to appear in this directory.