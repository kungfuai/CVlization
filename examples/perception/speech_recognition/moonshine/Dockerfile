FROM python:3.11-slim

WORKDIR /workspace

# Moonshine pulls ONNX / Keras-on-Torch and needs ffmpeg + libsndfile for
# audio decoding. Slim base keeps the image small (~the on-device pitch).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    KERAS_BACKEND=torch

CMD ["python", "predict.py"]
