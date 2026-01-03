FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set cache directories for model downloads
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

WORKDIR /workspace
