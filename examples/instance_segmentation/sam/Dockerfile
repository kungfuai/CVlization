FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y git \ 
    && rm -rf /var/lib/apt/lists/*
RUN pip install scikit-image==0.23.2 matplotlib==3.9.* kornia==0.7.2 segment-anything==1.0
RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git
RUN pip install timm==1.0.*
