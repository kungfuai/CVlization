FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# System dependencies for video IO and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip and install curated inference dependencies.
# Training-heavy extras (deepspeed/onnx/tensorrt) are omitted for a leaner image.
RUN python -m pip install --upgrade pip
RUN cat > /tmp/requirements.inference.txt <<'EOF' && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.inference.txt
torchvision>=0.19.0
accelerate>=1.1.1
diffusers==0.31.0
transformers>=4.49.0
tokenizers>=0.20.3
omegaconf
einops
tqdm
imageio
imageio-ffmpeg
opencv-python-headless
easydict
ftfy
numpy==1.24.4
av==13.1.0
pillow
sentencepiece
huggingface_hub[cli]
open_clip_torch
git+https://github.com/openai/CLIP.git
ml_collections
requests
scipy
scikit-image
pandas
decord
EOF

# Optional performance package; build continues if unavailable.
# Note: Flash attention kernels are already bundled with PyTorch 2.9.1 (SDPA),
# so we avoid installing flash-attn to prevent heavy source builds during image creation.
RUN pip install --no-cache-dir xformers || echo "xformers install skipped (optional)"

# Utilities sometimes imported at runtime
RUN pip install --no-cache-dir lmdb

# Make repo importable without installation
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Mount checkpoints and outputs at runtime
VOLUME ["/workspace/checkpoints", "/workspace/videos"]

CMD ["/bin/bash"]
