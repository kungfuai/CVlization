# Fun-CosyVoice3-0.5B: Zero-shot multilingual TTS inference
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    curl \
    wget \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clone CosyVoice at pinned commit for reproducibility
WORKDIR /opt
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git checkout 2d6bb9bd80aad0f46c4324dd0b5ee23a30b090a3 && \
    git submodule update --init --recursive

# Create optimized requirements (exclude tensorrt and deepspeed)
WORKDIR /opt/CosyVoice
RUN grep -v "tensorrt" requirements.txt | grep -v "deepspeed" > requirements_inference.txt

# Install pynini first
RUN pip install --no-cache-dir pynini==2.1.6.post1

# Install CosyVoice dependencies
RUN pip install --no-cache-dir -r requirements_inference.txt

# Set PYTHONPATH
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

WORKDIR /workspace

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/root/.cache/huggingface
ENV MODELSCOPE_CACHE=/root/.cache/modelscope

CMD ["python", "predict.py"]
