FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies (build-essential needed for PyAV)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    build-essential \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install xformers from cu128 index (matches base image PyTorch 2.9.1+cu128, includes Blackwell sm_120)
RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies (torch/torchvision/xformers already installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: confirm CUDA arch support
RUN python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('arch', torch.cuda.get_arch_list())"

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default command
CMD ["python", "predict.py", "--help"]
