FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install dependencies explicitly to keep torch 2.9.1 from the base image.
RUN pip install --no-cache-dir \
      openai==2.12.0 \
      transformers==4.57.2 \
      aiohttp \
      anthropic \
      blake3 \
      cachetools \
      cbor2 \
      cloudpickle \
      compressed-tensors==0.12.2 \
      depyf==0.20.0 \
      diskcache \
      einops \
      fastapi \
      flashinfer-python \
      gguf \
      lark==1.2.2 \
      llguidance \
      lm-format-enforcer==0.11.3 \
      "mistral_common>=1.8.5" \
      model-hosting-container-standards \
      msgspec \
      ninja \
      numba==0.61.2 \
      "openai-harmony>=0.0.3" \
      opencv-python-headless \
      outlines_core==0.2.11 \
      partial-json-parser \
      prometheus-fastapi-instrumentator \
      prometheus_client \
      protobuf \
      psutil \
      py-cpuinfo \
      pybase64 \
      "pydantic>=2.12.0" \
      python-json-logger \
      pyzmq \
      "ray>=2.48.0" \
      regex \
      scipy \
      sentencepiece \
      setproctitle \
      tiktoken \
      watchfiles \
      xgrammar \
    && pip install --no-cache-dir --no-deps vllm==0.12.0

COPY . .

CMD ["python3", "serve.py"]
