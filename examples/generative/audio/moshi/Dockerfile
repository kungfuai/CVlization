FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Moshi needs portaudio (sounddevice) and libsndfile (soundfile/sphn) even
# in headless containers -- the imports require the shared libs even when
# we only do file I/O. ffmpeg makes the bundled CVL sample (and arbitrary
# user audio) easier to munge if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

CMD ["python", "predict.py"]
