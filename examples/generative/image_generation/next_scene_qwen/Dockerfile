FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers and transformers
RUN pip install \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    huggingface-hub>=0.24.0 \
    safetensors>=0.4.4 \
    peft>=0.12.0 \
    sentencepiece \
    protobuf

# Install image processing dependencies
RUN pip install \
    pillow \
    opencv-python \
    imageio \
    matplotlib

# Install optional optimizations
RUN pip install ninja packaging wheel setuptools || true
RUN pip install flash-attn --no-build-isolation || echo "flash-attn installation skipped"

WORKDIR /workspace

# Set cache directories
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# Create output directory
RUN mkdir -p /workspace/outputs

CMD ["/bin/bash"]
