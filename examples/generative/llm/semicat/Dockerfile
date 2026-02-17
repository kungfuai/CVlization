FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y wget unzip gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (matching semicat's environment.yaml)
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install \
    torchmetrics==1.8.2 \
    lightning==2.6.0 \
    hydra-core==1.3.2 \
    hydra-optuna-sweeper==1.2.0 \
    hydra-colorlog==1.2.0 \
    rich==13.9.4 \
    rootutils==1.0.7 \
    einops==0.8.1 \
    datasets==4.5.0 \
    transformers==4.41.0 \
    jaxtyping==0.3.5 \
    timm==1.0.8 \
    wandb==0.19.1 \
    torchdiffeq==0.2.5

# Install flash-attention prebuilt wheel (cu128 + torch2.7 + cp311)
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl

ENV PYTHONPATH=/workspace/vendor
ENV PROJECT_ROOT=/workspace/vendor
