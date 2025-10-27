FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y git wget unzip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 # for imgaug \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install scikit-image==0.24.0 lightning==2.4.0
RUN pip install wandb==0.17.* pycocotools==2.0.8
RUN pip install albumentations==1.4.23