For containerized workflows, please refer to `../animate_diff_cog`.

### Install

```
pip install diffusers==0.24.0
<!-- pip install diffusers[torch]==0.11.1 -->
```

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/

bash download_bashscripts/0-MotionModule.sh
bash download_bashscripts/1-ToonYou.sh
```

### Quickstart

```
cd examples/video_gen/animate_diff
python predict.py --config configs/prompts/1-ToonYou.yaml
```

### Reference

https://github.com/talesofai/AnimateDiff/blob/server/animatediff/pipelines/pipeline_animation.py#L291