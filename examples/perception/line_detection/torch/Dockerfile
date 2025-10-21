FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace
RUN pip install scikit-image==0.19.3 pytorch-lightning==1.5.9
RUN pip install wandb==0.17.* pycocotools==2.0.7
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y wget unzip \ 
    && rm -rf /var/lib/apt/lists/*