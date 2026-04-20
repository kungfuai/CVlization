FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error

CMD ["python", "predict.py", "--help"]
