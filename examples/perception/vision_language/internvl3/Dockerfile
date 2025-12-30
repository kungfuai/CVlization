FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Set working directory
WORKDIR /workspace
