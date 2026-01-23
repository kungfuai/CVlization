# Video Artifact Removal - Training and Inference
# Base: PyTorch 2.9.1 with CUDA 12.8 runtime

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cvl-cache/torch
ENV HF_HOME=/cvl-cache/huggingface

# For dataset caching - maps to host ~/.cache/cvlization/
ENV XDG_CACHE_HOME=/cvl-cache

# Copy source code
COPY . .

# Default command
CMD ["/bin/bash"]
