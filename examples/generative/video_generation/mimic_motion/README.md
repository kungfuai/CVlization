Adapted from [MimicMotion](https://github.com/Tencent/MimicMotion)

## Quick Start

```bash
bash examples/generative/video_generation/mimic_motion/build.sh
# Optional: prefetch weights into ~/.cache/huggingface
bash examples/generative/video_generation/mimic_motion/download_models.sh

export HF_TOKEN=<your_huggingface_token>
bash examples/generative/video_generation/mimic_motion/predict.sh
# Optionally override inference steps, e.g. only 2 steps for smoke tests
# bash examples/generative/video_generation/mimic_motion/predict.sh --num_inference_steps 2
# Toggle slicing features from the CLI as needed
# bash examples/generative/video_generation/mimic_motion/predict.sh --attention_slicing off --vae_slicing off
# Reduce spatial/temporal load without editing YAML
# bash examples/generative_video_generation/mimic_motion/predict.sh --num_frames 24 --resolution 384 --sample_stride 4
```

## Run on your own data

Edit the `configs/test.yaml` file to specify the paths to your data. The default config is below:

```yaml
# base svd model path
base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1

# checkpoint path
ckpt_path: hf://tencent/MimicMotion/MimicMotion_1-1.pth

dwpose:
  det_path: hf://yzd-v/DWPose/yolox_l.onnx
  pose_path: hf://yzd-v/DWPose/dw-ll_ucoco_384.onnx

memory_optimization:
  cpu_offload: none  # options: none, model, sequential
  attention_slicing: true
  vae_slicing: true

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

## Model downloads & caching

- All large assets are fetched lazily at runtime through `hf://` URIs and cached under `$HF_HOME` (defaults to `~/.cache/huggingface`).  
- `download_models.sh` now simply pre-populates this shared cache and no longer writes duplicate copies into the example directory.
- The Docker `predict.sh` script mounts `${HOME}/.cache/huggingface` into the container so subsequent runs reuse the same weights.

## GPU memory tips

The upstream project reports that the 72-frame configuration can require **16 GB+ VRAM** and that the VAE decoder alone may demand up to 16 GB. This example enables a few diffusers optimizations by default:

- `attention_slicing` and `vae_slicing` reduce peak allocations during denoising and decoding.

Additional ways to reduce VRAM usage:

1. Lower `test_case[].num_frames`, `resolution`, or raise `sample_stride` to process fewer frames.
2. Set `memory_optimization.cpu_offload: model` (lighter offload) or `sequential` if you can tolerate slower runtimes; `sequential` is experimental because the upstream pipeline still performs explicit `.to(device)` calls.
3. Enable `memory_optimization.vae_cpu: true` for further savings at the cost of speed, or combine with `vae_tiling` via a custom config.
4. Use the CLI override `--num_inference_steps` to run with very small step counts (e.g., 2) when sanity-checking the pipeline.
5. Use `--attention_slicing on|off` and `--vae_slicing on|off` on the predict CLI to quickly toggle those memory optimizations without editing the YAML.
6. Combine `--num_frames`, `--resolution`, and `--sample_stride` overrides for rapid low-workload experiments.

Quantization is not yet wired into this example, but the pipeline uses float16 throughout. You can experiment with Diffusers `enable_sequential_cpu_offload`/`enable_model_cpu_offload` via the `memory_optimization` block, and the default Docker image now includes the `accelerate` package to support these modes.
