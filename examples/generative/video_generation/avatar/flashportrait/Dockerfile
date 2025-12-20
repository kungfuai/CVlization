FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install flash-attn from pre-built wheel (required for stable execution)
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Copy vendored code
COPY vendor/ /workspace/

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

CMD ["python", "fast_infer.py", "--help"]
