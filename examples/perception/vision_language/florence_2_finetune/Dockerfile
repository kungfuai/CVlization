FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    python3-dev \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY requirements.txt .
# Install flash-attn from wheel index (faster than building from source)
RUN /root/.local/bin/uv pip install --system \
    --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.6.3 \
    flash-attn==2.6.3
# Install remaining dependencies
RUN /root/.local/bin/uv pip install --system -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

CMD ["bash"]
