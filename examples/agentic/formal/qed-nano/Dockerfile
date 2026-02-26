FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    VLLM_LOGGING_LEVEL=ERROR

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "predict.py", "--help"]
