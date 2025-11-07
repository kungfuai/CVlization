# Note: Using CPU-only PyTorch base for smaller image size
# For GPU training, replace with: FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

CMD ["bash"]
