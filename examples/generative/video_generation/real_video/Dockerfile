FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# PyTorch 2.9+ has native Flash Attention via SDPA - no need for flash-attn package

# Set working directory
WORKDIR /workspace

# Copy vendored RealVideo modules (reproducible, no external dependency)
COPY vendor/RealVideo /workspace/RealVideo

# Install self_forcing module
RUN cd /workspace/RealVideo/self_forcing && pip install -e .

# Copy predict script
COPY predict.py /workspace/predict.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache

WORKDIR /workspace

# Default command
CMD ["python3", "predict.py", "--help"]
