FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install build tools required by Triton for JIT compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install vLLM from pre-built wheel (much faster than building from source)
# LightOnOCR requires vLLM >= 0.11.1
RUN pip install --no-cache-dir \
    vllm==0.11.2 \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

CMD ["bash"]
