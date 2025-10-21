###

This is still under development. There may be OOM issues when generating videos.

## Quick Start

In the root directory of the repo, run the following command to download the models:

```bash
bash examples/video_gen/wan/download_models.sh
```

To run a workflow, you can use the following command:

```bash
bash examples/video_gen/wan_comfy/predict.sh -p "a beautiful girl" -n "ugly, deformed, bad anatomy, bad hands, text, error, missing fingers, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad art, bad composition, distorted face, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature" -i examples/video_gen/animate_x/data/images/1.jpg -o output
```



## Reference

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [ComfyUI-Wan](https://comfyanonymous.github.io/ComfyUI_examples/wan/)

