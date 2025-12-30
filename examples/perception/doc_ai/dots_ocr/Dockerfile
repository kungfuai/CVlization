FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install base dependencies first (use versions compatible with PyTorch 2.6)
RUN /root/.local/bin/uv pip install --system accelerate pillow requests

# Install transformers 4.51.3 (specific version required by dots.ocr)
RUN pip install transformers==4.51.3

# Install qwen-vl-utils with decord for video support
RUN pip install "qwen-vl-utils[decord]"

# Install flash-attention from pre-built wheel (much faster than compiling)
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
