FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install \
    tiktoken==0.5.2 \
    einops==0.7.0 \
    wandb==0.16.*