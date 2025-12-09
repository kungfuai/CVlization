# OpenVLA + SimplerEnv Demo

Web-based demo for evaluating OpenVLA robot manipulation policies in ManiSkill2 simulation.

## Overview

This example provides a browser-based interface for running and visualizing OpenVLA policy inference on simulated robot manipulation tasks. The simulation runs on your GPU server, and frames are streamed to your browser via WebSocket.

**Supported tasks:**
- Google Robot: pick coke can, open/close drawer, move near, place in drawer
- WidowX: spoon on towel, carrot on plate, stack cube, eggplant in basket

## Requirements

- NVIDIA GPU with 24GB+ VRAM (A10, A100, RTX 4090, etc.)
- Docker with NVIDIA Container Toolkit
- CUDA 11.8+
- **Vulkan support** on host (required by SAPIEN simulator for GPU rendering)

### GPU and Vulkan Requirements

SimplerEnv uses the SAPIEN simulator which requires Vulkan for GPU-accelerated rendering. RTX-class GPUs are recommended. The host system must have:

1. NVIDIA driver with Vulkan ICD properly configured
2. The Vulkan ICD JSON (`/etc/vulkan/icd.d/nvidia_icd.json`) pointing to a library with Vulkan symbols

To verify Vulkan works on your host:
```bash
# Install vulkan-tools if needed
sudo apt-get install vulkan-tools
vulkaninfo | grep "GPU id"
```

If Vulkan is not working, you may need to install the full NVIDIA driver package (e.g., `libnvidia-gl-550`) rather than just the headless/compute driver.

**Note:** Running requires accepting the NVIDIA Omniverse EULA (handled automatically by scripts via `ACCEPT_EULA=Y`).

## Quick Start

```bash
# Build the Docker image
# Note: Base image is ~46GB, first pull takes significant time
./build.sh

# Run smoke test (verifies environment works)
./test.sh

# Start the demo server
./run.sh
```

Then open http://localhost:8000 in your browser.

## Usage

1. Select a task from the dropdown
2. (Optional) Check "Random actions" to test without loading the model
3. Click "Start Episode"
4. Watch the robot attempt the task
5. The episode ends when the task succeeds, truncates, or reaches max steps

## Architecture

```
┌─────────────────┐         WebSocket          ┌──────────────┐
│                 │ ────────────────────────── │              │
│  Browser UI     │        JSON + base64       │  GPU Server  │
│  (JavaScript)   │ ◀────── frames ──────────  │  (Python)    │
│                 │                            │              │
└─────────────────┘                            └──────────────┘
                                                     │
                                               ┌─────┴─────┐
                                               │           │
                                          ┌────┴───┐  ┌────┴───┐
                                          │OpenVLA │  │ManiSk. │
                                          │ Policy │  │  Sim   │
                                          └────────┘  └────────┘
```

## Files

- `server.py` - FastAPI + WebSocket server
- `sim_runner.py` - ManiSkill2 environment + OpenVLA policy loop
- `config.yaml` - Configuration (model path, max steps, frame rate)
- `static/` - Web UI (HTML, CSS, JavaScript)

## Configuration

Edit `config.yaml` to customize:

```yaml
model_path: "openvla/openvla-7b"  # HuggingFace model ID
max_steps: 200                     # Steps per episode
frame_delay_ms: 100                # Delay between frames (ms)
jpeg_quality: 80                   # Frame compression quality
```

## Remote Access

If running on a remote server:

```bash
# Option 1: SSH tunnel
ssh -L 8000:localhost:8000 user@your-server
# Then open http://localhost:8000

# Option 2: Direct access (if firewall allows)
# Open http://your-server:8000
```

## Troubleshooting

**"No GPU found" errors:**
- Ensure NVIDIA Container Toolkit is installed
- Run `nvidia-smi` to verify GPU is accessible
- Check `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`

**Vulkan/rendering errors:**
- The container uses EGL for headless rendering
- If issues persist, try setting `MUJOCO_GL=osmesa`

**Out of memory:**
- OpenVLA-7B requires ~14GB VRAM for inference
- Close other GPU processes or use a larger GPU

**Segmentation fault during Isaac Sim startup:**
- This is a known issue with certain GPU/driver combinations
- The base image (`qudelin/simpler-env:base`) uses Isaac Sim 2023.1 which may have compatibility issues with newer NVIDIA drivers (550.x)
- Workarounds:
  1. Try using an older NVIDIA driver (535.x or earlier)
  2. Try running on a different GPU type (A100, RTX 4090 are known to work)
  3. Check the [SimplerEnv issues](https://github.com/simpler-env/SimplerEnv/issues) for updates

## References

- [OpenVLA](https://github.com/openvla/openvla) - Open-source Vision-Language-Action model
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - Simulation evaluation for robot policies
- [ManiSkill2](https://github.com/haosulab/ManiSkill) - Robot manipulation simulation
- [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) - OpenVLA integration for SimplerEnv
