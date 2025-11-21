FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone Scope repository
WORKDIR /opt
RUN git clone https://github.com/daydreamlive/scope.git

# Install Scope dependencies manually first (excluding packages that need torch during build)
WORKDIR /opt/scope
RUN pip install \
    accelerate>=1.1.1 \
    aiortc>=1.13.0 \
    diffusers>=0.31.0 \
    easydict>=1.13 \
    einops>=0.8.1 \
    fastapi>=0.116.1 \
    ftfy>=6.3.1 \
    httpx>=0.28.1 \
    huggingface-hub>=0.26.0 \
    imageio \
    imageio-ffmpeg \
    lmdb>=1.7.3 \
    numpy>=2.0.0 \
    omegaconf>=2.3.0 \
    opencv-python-headless \
    peft>=0.17.1 \
    pillow>=11.0.0 \
    pydantic>=2.9.0 \
    python-multipart>=0.0.20 \
    pyyaml>=6.0.2 \
    safetensors>=0.4.5 \
    tokenizers>=0.21.0 \
    transformers>=4.46.0 \
    triton>=3.4.0 \
    uvicorn>=0.32.0 \
    twilio>=9.8.0

# Install packages that need torch during build (one at a time to avoid conflicts)
RUN pip install kernels>=0.10.4 || echo "kernels installation skipped (not critical)"
RUN pip install torchao>=0.13.0 || echo "torchao installation skipped (not critical)"
RUN pip install torchcodec>=0.7.0 || echo "torchcodec installation skipped (not critical)"
RUN pip install sageattention>=2.2.0 || echo "sageattention installation skipped (not critical)"

# Install Scope without dependencies (already installed above)
RUN pip install -e . --no-deps

# Patch Scope to make flash-attention optional
# This allows Scope to fall back to standard attention when flash-attn is not available
RUN python3 << 'EOF'
import re
attention_file = "/opt/scope/src/scope/core/pipelines/wan2_1/modules/attention.py"
with open(attention_file, "r") as f:
    content = f.read()
# Replace the flash_attn import with a try-except block
content = content.replace(
    "from flash_attn import flash_attn_func",
    """try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None"""
)
with open(attention_file, "w") as f:
    f.write(content)
print("Patched attention.py to make flash-attention optional")
EOF

# Skip flash-attn due to ABI incompatibility with PyTorch 2.6.0
# Scope will fall back to standard attention implementation
# RUN pip install ninja packaging wheel setuptools || true
# RUN pip install flash-attn==2.8.3 --no-build-isolation || echo "flash-attn installation skipped (not critical)"

# Set up workspace
WORKDIR /workspace

# Copy example files
COPY predict.py model_config.yaml /workspace/

# Set cache directories
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
ENV MODEL_FOLDER=/root/.cache/huggingface/wan_models

# Create necessary directories
RUN mkdir -p /workspace/outputs /root/.cache/huggingface/wan_models

CMD ["/bin/bash"]
