FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/flowrvs.requirements.txt

# Keep preinstalled CUDA/PyTorch stack from the base image; install remaining deps only.
RUN grep -Ev '^(torch==|torchvision==|triton==|nvidia-)' /tmp/flowrvs.requirements.txt > /tmp/flowrvs_filtered.txt \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/flowrvs_filtered.txt

WORKDIR /workspace
