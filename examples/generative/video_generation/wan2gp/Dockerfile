# Wan2GP (LTX2/Wan/LongCat) CLI example
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# Install flash-attn from prebuilt wheel (CUDA 12.8 + PyTorch 2.9 + Python 3.11)
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu128

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python", "predict.py", "--help"]
