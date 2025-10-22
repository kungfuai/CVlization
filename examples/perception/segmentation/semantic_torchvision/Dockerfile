FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y git wget unzip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 # for imgaug \ 
    && rm -rf /var/lib/apt/lists/*
RUN pip install scikit-image==0.19.3 pytorch-lightning==1.5.9
RUN pip install wandb==0.17.* pycocotools==2.0.7
RUN pip install imgaug==0.4.*