FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.5.1 --no-cache-dir
COPY requirements requirements/
RUN pip install -r requirements/framework.txt
RUN pip install groundingdino-py

# Install additional required packages
RUN pip install wan@git+https://github.com/Wan-Video/Wan2.1 && \
    pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps
RUN pip install sam2==1.1.0