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

# Upgrade to PyTorch 2.8.0 (or nightly if 2.8 not released) with CUDA 12.9 for Blackwell (sm_120) support
# Using cu129 for Blackwell GPU compatibility (RTX PRO 6000, RTX 50-series)
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 || \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Clone SAM3 repo at specific commit for reproducibility
RUN git clone https://github.com/facebookresearch/sam3 /opt/sam3 && \
    cd /opt/sam3 && \
    git checkout 84cc43bca4347b772f17d1078a1ddb4c054655c2

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface
ENV PYTHONPATH=/opt/sam3:$PYTHONPATH

WORKDIR /workspace

CMD ["bash"]
