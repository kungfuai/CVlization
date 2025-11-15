FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install torch first (needed during dependency build)
RUN /root/.local/bin/uv pip install --system torch==2.6.0 torchvision

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN /root/.local/bin/uv pip install --system --no-build-isolation -r requirements.txt

# Set working directory
WORKDIR /workspace
