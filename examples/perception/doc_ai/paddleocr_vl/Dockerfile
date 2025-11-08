FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PaddlePaddle GPU and PaddleOCR-VL
# PaddleOCR 3.x requires PaddlePaddle 3.x (3.2.0 for CUDA 12.6)
# Install custom safetensors wheel that supports PaddlePaddle framework
# Note: PaddlePaddle already includes Pillow and numpy, so no need to install separately
RUN pip install --no-cache-dir paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && \
    pip install --no-cache-dir "paddleocr[doc-parser]" && \
    pip install --no-cache-dir https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
