FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# NeMo needs Cython before pip-installing the asr extras, plus ffmpeg/libsndfile
# for audio decoding via librosa/soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir Cython==3.0.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    NEMO_DISABLE_LOGGING=1

CMD ["python", "predict.py"]
