# OpenVLA Single-Step Inference

Single-step inference demo for OpenVLA, visualizing predicted robot actions on static images.

## Overview

This example loads the OpenVLA-7B vision-language-action model and runs inference on input images to predict robot manipulation actions. No simulator required - just model inference with action visualization.

**What it does:**
- Takes an image + natural language instruction
- Predicts a 7-DoF robot action (position, orientation, gripper)
- Visualizes the action overlaid on the input image

**What it doesn't do:**
- No closed-loop control (single prediction only)
- No simulation or robot execution
- No sequential task completion

## Requirements

- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
- Docker with NVIDIA Container Toolkit
- ~14GB disk space for model weights (downloaded on first run)

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run basic tests
./test.sh

# Run interactive demo
./run.sh
```

## Usage

### Interactive Mode

```bash
./run.sh
```

This starts an interactive session where you can:
1. View sample images
2. Enter natural language instructions
3. See predicted actions visualized

### Single Image Inference

```bash
# With your own image
./run.sh --image /workspace/sample_images/robot_view.jpg \
         --instruction "pick up the red cup"

# Save output without display (headless)
./run.sh --image img.jpg --instruction "open drawer" \
         --output outputs/result.png --no-show
```

### Command Line Options

```
--image, -i       Path to input image
--instruction, -t Task instruction (default: "pick up the object")
--interactive     Run in interactive mode
--model           HuggingFace model path (default: openvla/openvla-7b)
--unnorm-key      Dataset key for action unnormalization (default: bridge_orig)
--output, -o      Output path for visualization
--no-show         Don't display visualization (headless mode)
--device          Device to run on (default: cuda:0)
```

## Output Format

The model predicts a 7-DoF action:

| Index | Name    | Description                    | Range      |
|-------|---------|--------------------------------|------------|
| 0     | dx      | End-effector X position delta  | [-1, 1]    |
| 1     | dy      | End-effector Y position delta  | [-1, 1]    |
| 2     | dz      | End-effector Z position delta  | [-1, 1]    |
| 3     | droll   | Orientation roll delta         | [-1, 1]    |
| 4     | dpitch  | Orientation pitch delta        | [-1, 1]    |
| 5     | dyaw    | Orientation yaw delta          | [-1, 1]    |
| 6     | gripper | Gripper command (0=close, 1=open) | [0, 1] |

## Adding Sample Images

Place robot camera images in the `sample_images/` directory:

```bash
sample_images/
├── kitchen_scene.jpg
├── tabletop_objects.jpg
└── drawer_view.jpg
```

Images should be:
- RGB format (JPG or PNG)
- Similar to robot camera viewpoint
- 256x256 pixels (will be resized if different)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Image    │────▶│    OpenVLA-7B    │────▶│  7-DoF Action   │
│  + Instruction  │     │  (14GB, bfloat16)│     │  Prediction     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Visualization  │
                                                │  (matplotlib)   │
                                                └─────────────────┘
```

## Performance

| GPU          | VRAM Usage | Inference Time |
|--------------|------------|----------------|
| L4 (24GB)    | ~14GB      | ~200ms/step    |
| A10 (24GB)   | ~14GB      | ~150ms/step    |
| A100 (40GB)  | ~14GB      | ~100ms/step    |

*With Flash Attention 2 enabled*

## Troubleshooting

**Out of memory:**
- OpenVLA-7B requires ~14GB VRAM
- Close other GPU processes
- Try `--device cpu` (very slow, ~30s/step)

**Model download fails:**
- Check internet connectivity
- Verify HuggingFace cache permissions
- Model weights are cached at `~/.cache/huggingface/`

**Visualization not showing:**
- Use `--output result.png` to save instead
- Add `--no-show` for headless servers

## Version Compatibility

This example uses pinned dependency versions as specified by the OpenVLA team:

| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.2.0 | Required |
| transformers | 4.40.1 | Required |
| tokenizers | 0.19.1 | Must match transformers |
| timm | 0.9.10 | Must be < 1.0.0 |
| flash-attn | 2.5.5 | Optional but recommended |

**Why not newer versions?**

OpenVLA's model code (loaded via `trust_remote_code=True`) has hard-coded version checks and API dependencies:
- Rejects `timm >= 1.0.0` with `NotImplementedError`
- Missing `_supports_sdpa` attribute required by transformers 4.50+
- Attention implementation tied to specific flash-attn API

Tested with PyTorch 2.9.1 + transformers 4.57 + timm 1.0.22 - fails at model load. See `Dockerfile.torch29` for the experimental configuration.

**Future upgrade path:**
1. Fork the OpenVLA model code from HuggingFace
2. Update version checks and add missing API attributes
3. Test inference outputs match original model
4. Consider contributing upstream if successful

## References

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [HuggingFace Model](https://huggingface.co/openvla/openvla-7b)
