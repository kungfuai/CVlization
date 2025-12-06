# OpenVLA + SimplerEnv: Robot Manipulation Task Selector Demo
#
# This Dockerfile builds an environment for evaluating OpenVLA policies
# in ManiSkill2 simulation with a web-based task selector interface.
#
# Base: qudelin/simpler-env (includes SAPIEN + ManiSkill2 + CUDA + Vulkan)
# Note: Base image is ~46GB and includes SimplerEnv and ManiSkill2_real2sim

FROM qudelin/simpler-env:base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Environment variables for headless rendering
ENV DISPLAY=""
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System dependencies (base image may have most, but ensure ffmpeg is present)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone SimplerEnv-OpenVLA to get the OpenVLA policy wrapper
# (The base simpler-env image has the core SimplerEnv, but we need the
# OpenVLA-specific policy code from the fork)
RUN git clone --depth 1 https://github.com/DelinQu/SimplerEnv-OpenVLA.git /opt/simpler_env_openvla

# Override pip index to use standard PyPI (base image uses Chinese mirror that may timeout)
RUN pip config set global.index-url https://pypi.org/simple && \
    pip config set global.trusted-host pypi.org

# Install OpenVLA dependencies (versions from openvla/openvla pyproject.toml)
# IMPORTANT: Use --no-deps to prevent overwriting base image's torch/numpy/torchvision
# which are required by Isaac Sim. The base image has compatible versions already.
RUN pip install --no-cache-dir --no-deps \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    accelerate==0.25.0 && \
    pip install --no-cache-dir \
    "safetensors>=0.3.3" \
    "huggingface_hub>=0.19.3,<1.0" \
    regex \
    tqdm \
    psutil

# Install flash-attention for faster inference (optional, falls back to standard attention)
RUN pip install --no-cache-dir flash-attn==2.6.1 --no-build-isolation 2>/dev/null || \
    echo "Note: Flash attention not installed, will use slower attention"

# Install web server and image processing dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add SimplerEnv-OpenVLA policies to Python path
ENV PYTHONPATH="/opt/simpler_env_openvla:${PYTHONPATH:-}"

# Expose web server port
EXPOSE 8000

# Default command: start the web server
CMD ["python", "server.py"]
