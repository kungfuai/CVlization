FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxcb1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# vLLM 0.14.0 supports both text LLMs and vision-language models (VLMs)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "serve.py"]
