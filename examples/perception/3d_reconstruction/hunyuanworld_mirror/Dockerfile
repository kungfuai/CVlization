FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Clone and install HunyuanWorld-Mirror
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git /opt/hunyuanworld-mirror && \
    cd /opt/hunyuanworld-mirror && \
    sed -i 's/--save_gs", action="store_true", default=True/--save_gs", action="store_true", default=False/' infer.py && \
    sed -i 's/--save_rendered", action="store_true", default=True/--save_rendered", action="store_true", default=False/' infer.py && \
    sed -i 's/--save_colmap", action="store_true", default=True/--save_colmap", action="store_true", default=False/' infer.py && \
    uv pip install --system -r requirements.txt

# Install gsplat - try latest version compatible with PyTorch 2.9
RUN uv pip install --system gsplat || \
    pip install --index-url https://docs.gsplat.studio/whl/pt29cu128 gsplat || \
    echo "Warning: gsplat installation failed, 3D Gaussian output will not be available"

# Install onnxruntime (needed for infer.py)
RUN uv pip install --system onnxruntime

# Copy and install additional requirements
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Add repo to Python path
ENV PYTHONPATH="/opt/hunyuanworld-mirror:${PYTHONPATH}"
