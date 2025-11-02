FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.26.4 torchvision==0.20.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
