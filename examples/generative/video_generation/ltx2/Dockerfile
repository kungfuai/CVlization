# LTX-2: Audio-Video Generation Model
# 19B parameter DiT model requiring significant VRAM
# FP8 mode recommended for ~24-32GB VRAM GPUs

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Reinstall torchaudio to ensure it matches torch version
RUN pip install --no-cache-dir torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Environment variables for memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default command
CMD ["python", "predict.py", "--help"]
