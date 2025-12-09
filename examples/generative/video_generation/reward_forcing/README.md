# Reward-Forcing (Inference, Dockerized)

Dockerized runner for the Reward-Forcing text-to-video model using the upstream repository. Built on `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` and trimmed to inference-focused dependencies.

## Build (cvl preset: `build`)

```bash
cd examples/generative/video_generation/reward_forcing
./build.sh  # CVL_IMAGE=reward_forcing:latest by default
```

## Run inference (cvl preset: `predict`)

```bash
./run_inference.sh  # defaults to image tag reward_forcing:latest (matches cvl run)
# override defaults
CKPT_DIR=/path/to/checkpoints OUTPUT_DIR=/path/to/videos DATA_PATH=prompts/single_prompt.txt NUM_FRAMES=21 HF_TOKEN=... ./run_inference.sh --use_ema

# one-off prompt (writes a temp prompt file in outputs/)
./run_inference.sh -p "A serene lake at sunrise with mist over the water."
PROMPT="A serene lake at sunrise with mist over the water." ./run_inference.sh
```

What the script does:
- uses centralized checkpoints by default (`~/.cache/cvlization/reward-forcing/checkpoints`) and standard Hugging Face cache (`~/.cache/huggingface`)
- mounts checkpoints, outputs, and Hugging Face cache into the container
- lazily downloads checkpoints via `huggingface_hub.snapshot_download` if `CHECKPOINT_PATH` is missing (default repo: `Wan-AI/Wan2.1-T2V-1.3B`; uses `HF_TOKEN` if set)
- runs `python inference.py` with default config/checkpoint/prompt file

Notes:
- `HF_TOKEN` is required if the Hugging Face repo is gated.
- Optional `xformers`/`flash-attn` install is best-effort; build succeeds if they fail.
- Training-only packages (Deepspeed/ONNX/TensorRT) are omitted to keep the image light for inference.
- CVL CLI: `cvl run reward-forcing build` then `cvl run reward-forcing predict -- --use_ema` (pass extra args after `--`). Image tag defaults to `reward_forcing:latest`.
