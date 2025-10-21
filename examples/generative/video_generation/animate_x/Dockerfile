FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install decord==0.6.0 einops==0.7.*
RUN pip install matplotlib==3.8.0 opencv-python==4.9.*
RUN pip install onnxruntime-gpu==1.18.*
RUN pip install oss2==2.18.4
RUN pip install simplejson==3.20.1 pynvml==12.0.0 easydict==1.13
RUN pip install xformers==0.0.20 rotary-embedding-torch==0.5.3
RUN pip install fairscale==0.4.13 open-clip-torch==2.24.0 imageio==2.33.1 imageio-ffmpeg==0.4.9
RUN pip install kornia==0.7.1
RUN pip install pytorch-lightning==2.1.3

WORKDIR /workspace