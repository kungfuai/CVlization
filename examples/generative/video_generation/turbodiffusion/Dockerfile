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

# Clone TurboDiffusion with submodules (CUTLASS)
RUN git clone --recursive https://github.com/thu-ml/TurboDiffusion.git /workspace/TurboDiffusion

WORKDIR /workspace/TurboDiffusion

# Patch setup.py to remove compute_120a (requires CUDA 12.8+) for CUDA 12.6 compatibility
RUN sed -i 's/"-gencode", "arch=compute_120a,code=sm_120a",//' setup.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    triton>=3.3.0 \
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

# Optional: Install SpargeAttn for SageSLA attention (best quality/speed tradeoff)
RUN MAX_JOBS=2 pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git || true

# Copy inference wrapper
COPY predict.py /workspace/predict.py

WORKDIR /workspace

# Create checkpoints directory (will be mounted or downloaded at runtime)
RUN mkdir -p /workspace/checkpoints /workspace/outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/TurboDiffusion:$PYTHONPATH
