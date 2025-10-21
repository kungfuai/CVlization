FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install decord==0.6.0 einops==0.7.*
RUN pip install matplotlib==3.8.0 opencv-python==4.9.*
RUN pip install onnxruntime-gpu==1.18.*
RUN pip install omegaconf==2.3.0 cog==0.13.7
RUN pip install diffusers==0.32.2 transformers==4.48.3

WORKDIR /workspace