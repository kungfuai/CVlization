FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies including poppler and fonts required by olmOCR
RUN apt-get update && apt-get install -y \
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

# Install olmOCR with GPU support
# Using torch 2.5.1 which is compatible with cu124
RUN pip install --no-cache-dir \
    "olmocr[gpu]" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
