FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV MAX_JOBS=8
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV NINJA_NUM_JOBS=8

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Install Block-Sparse-Attention (required for LCSA)
WORKDIR /opt
RUN git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention.git
WORKDIR /opt/Block-Sparse-Attention
RUN pip install --no-cache-dir packaging ninja \
    && TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
       MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 NINJA_NUM_JOBS=8 \
       bash -lc 'if [ -n "${BLOCK_SPARSE_ATTN_WHEEL:-}" ]; then \
           pip install "${BLOCK_SPARSE_ATTN_WHEEL}"; \
         else \
           pip install . --no-build-isolation; \
         fi'

WORKDIR /workspace
