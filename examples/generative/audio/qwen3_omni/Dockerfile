FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Audio decode libs (librosa/soundfile/av need libsndfile + ffmpeg).
# ffmpeg is explicitly required by qwen-omni-utils' video path.
# build-essential because transformers can JIT-compile triton kernels on
# first run and the runtime base lacks gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_VERBOSITY=error

CMD ["python", "predict.py"]
