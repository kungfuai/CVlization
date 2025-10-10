FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements
COPY requirements.txt .

# Install base dependencies first
RUN /root/.local/bin/uv pip install --system torch==2.5.1 torchvision==0.20.1 accelerate pillow requests

# Install transformers 4.51.3 (specific version required by dots.ocr)
RUN pip install transformers==4.51.3

# Install qwen-vl-utils with decord for video support
RUN pip install "qwen-vl-utils[decord]"

# Install flash-attention separately with pip (uv has issues with build isolation)
RUN pip install flash-attn --no-build-isolation

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
