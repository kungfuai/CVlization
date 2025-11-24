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

# Clone MASt3R with submodules (includes DUSt3R)
RUN git clone --recursive https://github.com/naver/maSt3R.git /opt/mast3r && \
    cd /opt/mast3r

# Install DUSt3R dependencies first (from submodule)
WORKDIR /opt/mast3r/dust3r
RUN uv pip install --system -r requirements.txt

# Compile CUDA kernels for RoPE (optional but recommended for performance)
# If compilation fails (e.g., on newer GPU architectures), inference will still work
RUN cd /opt/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace || echo "Warning: RoPE CUDA kernel compilation failed, will use fallback implementation" && \
    cd /workspace

# Install MASt3R requirements
WORKDIR /opt/mast3r
RUN uv pip install --system -r requirements.txt

# Copy and install additional requirements
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system -r /workspace/requirements.txt

# Copy inference script
COPY predict.py /workspace/predict.py
RUN chmod +x /workspace/predict.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Add mast3r and dust3r to Python path
ENV PYTHONPATH="/opt/mast3r:/opt/mast3r/dust3r:${PYTHONPATH}"

WORKDIR /workspace
