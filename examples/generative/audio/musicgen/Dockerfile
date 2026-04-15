# AudioCraft MusicGen inference
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

RUN git clone https://github.com/facebookresearch/audiocraft.git /opt/audiocraft && \
    cd /opt/audiocraft && \
    git checkout 896ec7c && \
    python -m pip install --no-cache-dir --no-deps -e .

CMD ["python", "predict.py"]
