FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    git+https://github.com/huggingface/diffusers \
    transformers>=4.51.3 \
    accelerate>=0.33.0 \
    huggingface-hub>=0.24.0 \
    safetensors>=0.4.4 \
    sentencepiece \
    protobuf

RUN pip install \
    gradio>=4.0.0 \
    pillow \
    numpy \
    python-pptx \
    timm

WORKDIR /workspace

ENV HF_HOME=/cvl-cache/huggingface
ENV TRANSFORMERS_CACHE=/cvl-cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/cvl-cache/huggingface/datasets

RUN mkdir -p /workspace/outputs

CMD ["/bin/bash"]
