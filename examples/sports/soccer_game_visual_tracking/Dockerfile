FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install transformers==4.40.* opencv-python==4.10.* umap-learn==0.5.6 scikit-learn==1.5.* tqdm \
    sentencepiece protobuf supervision==0.23.* ultralytics==8.2.* gdown

