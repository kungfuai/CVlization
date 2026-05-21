# MolmoAct2 Vision-Language-Action Inference
#
# 5B-parameter VLA model from Allen AI for robot manipulation.
# Uses flow-matching continuous action expert with optional
# depth-token reasoning (Think variant).

# Use PyTorch 2.10+cu128 base for Blackwell GPU support (sm_120)
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Core dependencies (einops and requests required by MolmoAct2 custom code)
# Ubuntu 24 requires --break-system-packages since pip is managed by the OS
RUN pip install --no-cache-dir --break-system-packages \
    "transformers>=4.48" \
    accelerate \
    einops \
    requests \
    pillow \
    numpy \
    huggingface_hub \
    scipy \
    sentencepiece \
    protobuf

# cvlization is mounted at /cvlization_repo via predict.sh (PYTHONPATH=/cvlization_repo)

WORKDIR /workspace

# predict.py is mounted from the example directory; don't COPY it to keep the
# image decoupled from example code updates.

# Environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "predict.py", "--help"]
