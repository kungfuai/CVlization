FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
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

# Clone MonST3R with submodules (croco, viser)
RUN git clone --recursive https://github.com/junyi42/monst3r.git /opt/monst3r && \
    cd /opt/monst3r

# Install Python dependencies
WORKDIR /opt/monst3r
RUN uv pip install --system torch torchvision && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system imageio

# Install gdown for checkpoint downloads (will happen at runtime)
RUN uv pip install --system gdown

# Create checkpoint directories (will be populated at runtime)
RUN mkdir -p third_party/sam2/checkpoints && \
    mkdir -p third_party/RAFT/models && \
    mkdir -p checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Add monst3r to Python path
ENV PYTHONPATH="/opt/monst3r:${PYTHONPATH}"

WORKDIR /workspace
