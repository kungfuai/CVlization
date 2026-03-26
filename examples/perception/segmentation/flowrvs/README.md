# FlowRVS (Referring Video Segmentation)

FlowRVS segments video objects described by natural-language prompts. It is built on Wan2.1 flow-matching components and achieves state-of-the-art results on MeViS.

## Sample

**Input** — basketball clip (auto-downloaded from `zzsi/cvl`):

![input](https://huggingface.co/datasets/zzsi/cvl/resolve/main/flowrvs/sample_basketball_input.gif)

Prompts: `"the man wearing colorful shoes shoots the ball"` · `"the man who is defending"` · `"basketball"`

**Output** — color-coded segmentation mask overlay per prompt:

![output](https://huggingface.co/datasets/zzsi/cvl/resolve/main/flowrvs/sample_basketball_output.gif)

## What to Expect

- **First run**: downloads ~10GB (Wan2.1-T2V-1.3B base model) + ~5GB (FlowRVS checkpoints), cached to `~/.cache/huggingface/` afterward
- **What it does**: segments one or more objects per text prompt, producing a color-coded mask overlay video
- **Output location**: saved to `outputs/flowrvs_result.mp4` in your current working directory
- **Output format**: MP4 overlay video at the input resolution; each prompt gets a distinct color mask
- **Runtime**: ~45s on an A100/H100 class GPU; requires ~33GB VRAM

## Requirements

- NVIDIA GPU with >= 33GB VRAM
- Docker with NVIDIA runtime
- Internet access for model downloads

## Build

```bash
bash examples/perception/segmentation/flowrvs/build.sh
```

## Run

No-arg default (basketball sample, three prompts):

```bash
cvl run flowrvs predict
```

Custom video and prompts:

```bash
bash examples/perception/segmentation/flowrvs/predict.sh \
  --input path/to/video.mp4 \
  --prompts "the person in red" "the dog" \
  --output outputs/result.mp4
```

Built-in samples: `--input basketball` (default) or `--input sample` (Ultraman clip).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `basketball` | Video path or built-in sample name |
| `--prompts` | sample-specific | One or more referring text prompts |
| `--output` | `outputs/flowrvs_result.mp4` | Output overlay video path |
| `--fps` | `12` | FPS for video decoding |
| `--height` | `480` | Inference height |
| `--width` | `832` | Inference width |
| `--dry-run` | — | Resolve paths and download weights, then exit |

## References

- FlowRVS repo: https://github.com/xmz111/FlowRVS
- FlowRVS paper: https://arxiv.org/abs/2510.06139
- FlowRVS weights: https://huggingface.co/xmz111/FlowRVS
