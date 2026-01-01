FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

# Install flash-attn from pre-built wheel (CUDA 12.8 + PyTorch 2.9 + Python 3.11)
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

COPY requirements requirements/
# Install requirements, skipping torch pin and flash_attn (already installed)
RUN grep -v -E "^(torch==|flash_attn)" requirements/framework.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt
RUN pip install groundingdino-py

# Install additional required packages
RUN pip install wan@git+https://github.com/Wan-Video/Wan2.1 && \
    pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps
RUN pip install sam2==1.1.0