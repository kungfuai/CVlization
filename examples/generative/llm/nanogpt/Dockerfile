FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install \
    tiktoken==0.8.0 \
    einops==0.8.0 \
    wandb==0.19.1