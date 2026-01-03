FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git

WORKDIR /opt/Wan2.2

# Install base requirements (excluding flash_attn which we install separately)
RUN pip install --upgrade pip setuptools wheel
RUN grep -v '^flash_attn' requirements.txt > /tmp/requirements-no-flash.txt
RUN pip install -r /tmp/requirements-no-flash.txt

# Install S2V-specific requirements (CosyVoice TTS, audio processing)
RUN pip install -r requirements_s2v.txt

# Pin transformers and peft versions for compatibility
RUN pip install --no-deps transformers==4.52.3 peft==0.17.1

# Install flash_attn from pre-built wheel (much faster than compiling)
# Source: https://flashattn.dev/
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Additional utilities
RUN pip install moviepy librosa

WORKDIR /workspace
