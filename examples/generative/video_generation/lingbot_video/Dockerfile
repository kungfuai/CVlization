# LingBot-Video: MoE Video Foundation Model
# Dense 1.3B: ~8GB VRAM (bf16)
# MoE 30B-A3B: ~24GB+ VRAM (~3B active params)

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

# Install lingbot-video package from GitHub (provides pipeline, transformer, scheduler)
RUN pip install --no-cache-dir "lingbot-video @ git+https://github.com/Robbyant/lingbot-video.git"

# Default command
CMD ["python", "predict.py", "--help"]
