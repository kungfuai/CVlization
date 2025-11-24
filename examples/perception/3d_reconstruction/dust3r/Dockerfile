FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Clone and install DUSt3R
RUN git clone --recursive https://github.com/naver/dust3r.git /opt/dust3r && \
    cd /opt/dust3r && \
    uv pip install --system -r requirements.txt

# Compile CUDA kernels for RoPE (optional but recommended for performance)
# If compilation fails (e.g., on newer GPU architectures), inference will still work
RUN cd /opt/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace || echo "Warning: RoPE CUDA kernel compilation failed, will use fallback implementation" && \
    cd /workspace

# Copy and install additional requirements
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Add repo to Python path
ENV PYTHONPATH="/opt/dust3r:${PYTHONPATH}"
