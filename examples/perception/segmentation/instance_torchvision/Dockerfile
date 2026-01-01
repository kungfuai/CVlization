FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y wget unzip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install scikit-image==0.24.0 lightning==2.4.0
RUN pip install wandb==0.17.* pycocotools==2.0.8
