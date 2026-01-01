FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install decord==0.6.0 einops==0.8.1
RUN pip install matplotlib==3.9.4 opencv-python==4.10.0.84
RUN pip install onnxruntime-gpu==1.22.0
RUN pip install omegaconf==2.3.0 cog==0.13.7
RUN pip install diffusers==0.36.0 transformers==4.48.3 accelerate==1.7.0

WORKDIR /workspace
