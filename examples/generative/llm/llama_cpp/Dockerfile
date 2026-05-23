# Use upstream's official server-cuda image: ships llama-server, llama-cli,
# llama-bench, llama-quantize prebuilt with CUDA support. Pinned to a known
# build for reproducibility (override with --build-arg LLAMA_CPP_TAG=...).
ARG LLAMA_CPP_TAG=full-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_TAG}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/app:$PATH \
    LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

WORKDIR /workspace

# llama.cpp's CUDA image is built on a slim ubuntu base; add Python and a few
# libs we need for the predict/serve wrappers (OpenAI client + image loader).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ca-certificates \
        libxcb1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# llama.cpp's `-hf` downloads to its own cache by default. Point it at the
# standard HF cache so weights are shared with vllm / sglang / transformers
# examples.
ENV LLAMA_CACHE=/root/.cache/huggingface/llama_cpp \
    HF_HOME=/root/.cache/huggingface

COPY . .

# Clear upstream's tool-wrapper ENTRYPOINT so our python wrappers (and
# any arbitrary docker run command) are executed directly.
ENTRYPOINT []

# Default to serving an OpenAI-compatible HTTP endpoint. Overridable in
# predict.sh / serve.sh which exec their own python wrapper.
CMD ["python3", "serve.py"]
