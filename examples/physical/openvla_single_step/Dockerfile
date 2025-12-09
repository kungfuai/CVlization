# OpenVLA Single-Step Inference
#
# Minimal Docker image for running OpenVLA inference on static images.
# No simulator required - just model inference with action visualization.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# Install numpy first (must be <2.0 for PyTorch compatibility)
RUN pip install --no-cache-dir "numpy==1.24.4"

# Install PyTorch with CUDA support (exact version per OpenVLA requirements)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install OpenVLA dependencies (exact versions per model requirements)
# See: https://github.com/openvla/openvla#installation
RUN pip install --no-cache-dir \
    "transformers==4.40.1" \
    "tokenizers==0.19.1" \
    accelerate \
    pillow \
    matplotlib \
    "timm==0.9.10" \
    sentencepiece \
    protobuf

# Install flash-attn for faster inference (exact version per OpenVLA requirements)
RUN pip install --no-cache-dir "flash-attn==2.5.5" --no-build-isolation || \
    echo "Flash attention not installed - will use standard attention"

WORKDIR /workspace

# Copy application files
COPY inference.py .
COPY sample_images/ ./sample_images/

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/workspace/.cache/huggingface

# Default command
CMD ["python", "inference.py", "--help"]
