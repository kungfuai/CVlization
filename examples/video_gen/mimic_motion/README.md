Adapted from [MimicMotion](https://github.com/Tencent/MimicMotion)

## Quick Start

```bash
bash examples/video_gen/mimic_motion/build.sh
bash examples/video_gen/mimic_motion/download_models.sh

export HF_TOKEN=<your_huggingface_token>
bash examples/video_gen/mimic_motion/predict.sh
```

## Run on your own data

Edit the `configs/test.yaml` file to specify the paths to your data. The default config is below:

```yaml
# base svd model path
base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1

# checkpoint path
ckpt_path: models/MimicMotion_1-1.pth

test_case:
  - ref_video_path: example_data/videos/pose1.mp4  # path to your reference video
    ref_image_path: example_data/images/demo1.jpg  # path to your reference image
    num_frames: 72  # number of frames in the reference video
    resolution: 576  # resolution of the reference video
    frames_overlap: 6  # number of frames to overlap between tiles
    num_inference_steps: 25  # number of inference steps
    noise_aug_strength: 0  # noise augmentation strength
    guidance_scale: 2.0  # guidance scale
    sample_stride: 2  # stride for sampling frames
    fps: 15  # frames per second
    seed: 42  # random seed
```
