FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python \
    pillow \
    requests \
    pycocotools \
    wandb \
    'git+https://github.com/cocodataset/panopticapi.git' \
    'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /workspace
