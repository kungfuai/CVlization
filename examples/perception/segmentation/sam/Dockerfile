FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y git wget unzip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install scikit-image==0.24.0 matplotlib==3.9.4 kornia==0.7.4 segment-anything==1.0
RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git
RUN pip install timm==1.0.15
