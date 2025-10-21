# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

# Install system dependencies
#RUN apt-get update && apt-get install -y \
#    curl \
#    git \
#    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install PyTorch nightly for FlexAttention support
# Using latest nightly build with CUDA 12.1
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

# Additional dependencies that might be useful
RUN pip install wandb==0.22.*

# Copy the rest of the code
COPY . /workspace/

# Set environment variables for multi-GPU training
ENV PYTHONUNBUFFERED=1
ENV NCCL_P2P_DISABLE=1
# Don't set CUDA_VISIBLE_DEVICES here - let train.sh control it

CMD ["bash"]
ENTRYPOINT []
