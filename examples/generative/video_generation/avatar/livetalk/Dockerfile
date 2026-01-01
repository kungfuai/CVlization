FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone LiveTalk and OmniAvatar
RUN git clone --depth=1 https://github.com/GAIR-NLP/LiveTalk /workspace/LiveTalk
RUN git clone --depth=1 https://github.com/Omni-Avatar/OmniAvatar /workspace/LiveTalk/OmniAvatar

# Apply OmniAvatar patches
WORKDIR /workspace/LiveTalk
RUN bash scripts/add_patch.sh

# Install requirements
COPY requirements.txt /workspace/LiveTalk/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install flash-attention (prebuilt wheel for torch 2.9 + cu128 + Python 3.11)
RUN python -m pip uninstall -y flash-attn flash_attn || true
RUN python -m pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.6.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Install LiveTalk package
RUN python setup.py develop

COPY predict.py /workspace/predict.py
COPY patches/ /workspace/patches/

ENV PYTHONUNBUFFERED=1
ENV MODEL_BASE=/workspace/LiveTalk/pretrained_checkpoints

WORKDIR /workspace
CMD ["python3", "/workspace/predict.py", "--help"]
