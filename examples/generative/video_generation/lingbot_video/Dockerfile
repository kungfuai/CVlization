# LingBot-Video: MoE Video Foundation Model
# Dense 1.3B: ~8GB VRAM (bf16)
# MoE 30B-A3B: ~80GB VRAM at 81 frames 832x480 (~3B active params)
#
# Upstream pinned to commit a638721 for reproducibility.
# Tested dependency stack (resolved in this image):
#   torch 2.7.0+cu128, diffusers 0.39.0, transformers 5.13.1,
#   flash-attn 2.8.3.post1 (prebuilt wheel, standard attention backend)
#
# Note: --batch-cfg requires flash_attn_3 (Hopper/Blackwell flash_attn_interface).
# Prebuilt wheels (sm80/sm90a/sm100a) do not cover sm120 (Blackwell Max-Q).
# Omitted until a compatible wheel is available; sequential CFG works correctly.

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install prebuilt FlashAttention v2 (standard attention backend)
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install lingbot-video package from GitHub, pinned to tested commit
RUN pip install --no-cache-dir \
    "lingbot-video @ git+https://github.com/Robbyant/lingbot-video.git@a638721cf2271804d02738b69f2ad788c4a559fc"

# Default command
CMD ["python", "predict.py", "--help"]
