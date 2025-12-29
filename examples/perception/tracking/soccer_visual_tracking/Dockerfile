FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install transformers==4.57.3 opencv-python==4.10.0.84 umap-learn==0.5.6 scikit-learn==1.5.2 tqdm \
    sentencepiece protobuf supervision==0.23.0 ultralytics==8.2.103 gdown==5.2.0

