FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

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

# Install Python dependencies (torch 2.9.1 already included in base image)
RUN /root/.local/bin/uv pip install --system transformers==4.57.3 pillow==11.1.0 requests==2.32.3 accelerate==1.2.1 einops==0.8.1

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
