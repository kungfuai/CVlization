# MolmoAct2 Vision-Language-Action Inference
#
# 5B-parameter VLA model from Allen AI for robot manipulation.
# Uses flow-matching continuous action expert with optional
# depth-token reasoning (Think variant).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
RUN pip install --no-cache-dir \
    "transformers>=4.48" \
    accelerate \
    pillow \
    numpy \
    huggingface_hub \
    scipy \
    sentencepiece \
    protobuf

WORKDIR /workspace

COPY inference.py .

# Environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "inference.py", "--help"]
