# LingBot-Video: MoE Video Foundation Model
# Dense 1.3B: ~8GB VRAM (bf16)
# MoE 30B-A3B: ~80GB VRAM at 81 frames 832x480 (~3B active params)
#
# Upstream pinned to commit a638721 for reproducibility.
# Tested dependency stack (resolved in this image):
#   torch 2.7.0+cu128, diffusers 0.39.0, transformers 5.13.1
#
# FlashAttention is NOT installed. The wrapper (predict.py) uses PyTorch native
# SDPA for sequential CFG attention — no flash_attn import is needed. The
# upstream runner (lingbot_video.runner) monkey-patches attn_implementation to
# flash_attention_3 by default, but this can be overridden with the env var
# LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa (no FA wheel required).
#
# --batch-cfg requires flash_attn_3 (Hopper/Blackwell flash_attn_interface).
# No prebuilt wheel covers sm120 (Blackwell Max-Q).

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

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install lingbot-video package from GitHub, pinned to tested commit
RUN pip install --no-cache-dir \
    "lingbot-video @ git+https://github.com/Robbyant/lingbot-video.git@a638721cf2271804d02738b69f2ad788c4a559fc"

# Default command
CMD ["python", "predict.py", "--help"]
