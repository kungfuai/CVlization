FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install build tools and runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libxcb1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Install vLLM (v0.14.0 has Transformers v5 support)
RUN pip install --no-cache-dir \
    vllm==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install transformers from git (LightOnOCR-2 requires unreleased features)
# Pinned to specific commit for reproducibility
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers@3d1264c7f4c63c9a83dcb14a07ff67f004c3fb77

# Install additional dependencies (pinned versions)
RUN pip install --no-cache-dir \
    pillow==11.1.0 \
    pypdfium2==4.30.1 \
    requests==2.32.3

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

CMD ["bash"]
