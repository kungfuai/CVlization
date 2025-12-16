# TurboDiffusion: 100-205x accelerated video generation
# Uses Wan2.1-1.3B model for text-to-video generation
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy TurboDiffusion source (included in example folder)
COPY ops/ /workspace/turbodiffusion/ops/
COPY rcm/ /workspace/turbodiffusion/rcm/
COPY imaginaire/ /workspace/turbodiffusion/imaginaire/
COPY SLA/ /workspace/turbodiffusion/SLA/
COPY inference/ /workspace/turbodiffusion/inference/
COPY setup.py pyproject.toml __init__.py /workspace/turbodiffusion/

# Download CUTLASS v3.4.1 headers
RUN git clone --depth 1 --branch v3.4.1 https://github.com/NVIDIA/cutlass.git \
    /workspace/turbodiffusion/ops/cutlass

WORKDIR /workspace/turbodiffusion

# Install Python dependencies
RUN pip install --no-cache-dir \
    "triton>=3.3.0" \
    einops \
    numpy \
    pillow \
    loguru \
    "imageio[ffmpeg]" \
    pandas \
    PyYAML \
    omegaconf \
    attrs \
    fvcore \
    ftfy \
    regex \
    transformers \
    nvidia-ml-py \
    huggingface_hub

# Install flash-attn from pre-built wheel (avoids OOM during compilation)
# Wheel from: https://github.com/mjun0812/flash-attention-prebuild-wheels
RUN pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.6.3%2Bcu126torch2.9-cp311-cp311-linux_x86_64.whl"

# Build and install TurboDiffusion CUDA extensions
RUN MAX_JOBS=2 pip install -e . --no-build-isolation

# Note: SpargeAttn (for SageSLA) has compatibility issues on RTX 30xx GPUs.
# The default 'sla' attention type works on all GPUs and provides similar speedup.

# Copy inference wrapper
COPY predict.py /workspace/predict.py

WORKDIR /workspace

# Create checkpoints directory
RUN mkdir -p /workspace/checkpoints /workspace/outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/turbodiffusion
