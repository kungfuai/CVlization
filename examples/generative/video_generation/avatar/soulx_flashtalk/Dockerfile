# Base image provides torch 2.7.1 + CUDA 12.8 (the versions SoulX-FlashTalk
# and xformers 0.0.31 are pinned against). Devel variant includes nvcc.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# System dependencies. ffmpeg is required: SoulX-FlashTalk muxes the generated
# (silent) video with the input audio via an ffmpeg subprocess.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install flash-attn from a prebuilt wheel (built for cu12 / torch 2.7 / cp311)
# before the other requirements so that xfuser sees the dependency satisfied
# and does not attempt to build flash-attn from source.
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Python dependencies (frozen versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace

# Clone the SoulX-FlashTalk repository, pinned to a known-good commit.
RUN git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git /workspace/SoulX-FlashTalk && \
    cd /workspace/SoulX-FlashTalk && \
    git checkout f54dc79511e001472a4e46e14179d0db981c1caf

COPY predict.py /workspace/predict.py

ENV PYTHONUNBUFFERED=1
ENV SOULX_DIR=/workspace/SoulX-FlashTalk
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace
CMD ["python3", "predict.py", "--help"]
