FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Upgrade PyTorch with CUDA 12.9 for Blackwell (sm_120) support
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 || \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

CMD ["bash"]
