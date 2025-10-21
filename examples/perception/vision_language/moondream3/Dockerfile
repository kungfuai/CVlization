FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

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

# Install PyTorch 2.8.0 for flex_attention support (required by Moondream3)
# Use CUDA 12.6 build (compatible with our CUDA 12.4 runtime)
# Pinned version for reproducibility
RUN pip install --upgrade torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install other Python dependencies
RUN /root/.local/bin/uv pip install --system transformers>=4.40.0 pillow>=10.0.0 requests>=2.31.0 accelerate>=0.34.0 einops>=0.7.0

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
