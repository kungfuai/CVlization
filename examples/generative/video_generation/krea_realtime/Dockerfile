FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone Krea Realtime SDK
WORKDIR /opt
RUN git clone https://github.com/krea-ai/realtime-video krea-sdk

# Patch SDK for PyTorch 2.5 compatibility (recompile_limit doesn't exist in 2.5)
WORKDIR /opt/krea-sdk
RUN sed -i '56s/^/# /' release_server.py

# Install uv for fast package installation
RUN pip install uv

# Copy SDK requirements file
COPY sdk_requirements.txt /tmp/sdk_requirements.txt

# Install all SDK dependencies using uv (much faster than pip)
# This includes 217 dependencies from the Krea SDK
WORKDIR /opt/krea-sdk
RUN uv pip install --system -r /tmp/sdk_requirements.txt

# Install flash-attn (optional, may fail on some systems)
RUN pip install ninja packaging wheel setuptools || true
RUN pip install flash-attn --no-build-isolation || echo "flash-attn installation skipped (not critical)"

# Set up workspace
WORKDIR /workspace

# Add Krea SDK to Python path
ENV PYTHONPATH=/opt/krea-sdk:$PYTHONPATH

# Set cache directories
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
ENV MODEL_FOLDER=/root/.cache/huggingface/wan_models

# Create necessary directories
RUN mkdir -p /workspace/outputs /root/.cache/huggingface/wan_models

CMD ["/bin/bash"]
