FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

# Install python dependencies (inference only)
# Note: SAM3 requires transformers from git; PyPI 4.57.3 (latest as of 2025-12-30) doesn't include Sam3Model yet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vendored SAM3 source (from github.com/facebookresearch/sam3, commit 2ec3c07)
COPY vendor/sam3 /opt/sam3
RUN cd /opt/sam3 && pip install --no-cache-dir -e .

# Make SAM3 importable everywhere
ENV PYTHONPATH=/opt/sam3

CMD ["bash"]
