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
RUN pip install --no-cache-dir -r requirements.txt
# vLLM 0.19.0 added Gemma 4 support but still declares transformers<5 in pip metadata.
# Install transformers 5.5.0 after vllm to override that constraint — this is the
# same approach used by the official vllm/vllm-openai:gemma4 image.
# vLLM 0.19.0 includes explicit transformers v5 compatibility fixes for OLMo-3,
# LFM2, DeepSeekVL2, Pixtral, and other models (PRs #37681, #38127, #38090, #38247).
RUN pip install --no-cache-dir transformers==5.5.0

COPY . .

CMD ["python3", "serve.py"]
