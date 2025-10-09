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

# Install PyTorch nightly for flex_attention support (required by Moondream3)
RUN /root/.local/bin/uv pip install --system --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# Install other Python dependencies
RUN /root/.local/bin/uv pip install --system transformers>=4.40.0 pillow>=10.0.0 requests>=2.31.0 accelerate>=0.34.0 einops>=0.7.0

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
