# PyTorch 2.9.1 with CUDA 12.8 for Blackwell GPU support
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies including poppler and fonts required by olmOCR
# Pre-accept Microsoft fonts EULA to avoid interactive prompt
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    poppler-utils \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    && rm -rf /var/lib/apt/lists/*

# Install olmOCR with GPU support and qwen-vl-utils for direct model inference
RUN pip install --no-cache-dir \
    "olmocr[gpu]" \
    qwen-vl-utils \
    accelerate \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
