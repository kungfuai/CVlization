FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Clone and install map-anything
RUN git clone https://github.com/facebookresearch/map-anything.git /opt/map-anything && \
    cd /opt/map-anything && \
    uv pip install --system -e .

# Copy requirements for additional dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface
