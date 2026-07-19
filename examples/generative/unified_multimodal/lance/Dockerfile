FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Lance dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install flash-attn from prebuilt wheel (cu128 + torch2.9, cp311)
# Required by Lance qwen2_navit.py (hard dependency on flash_attn_varlen_func)
RUN pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl"

# Build-time validation: ensure flash_attn and torch are importable
RUN python -c "import flash_attn, torch; print(flash_attn.__version__, torch.__version__)"

# Pin transformers to the version Lance requires (4.49.0 avoids LM loading bug)
RUN pip install --no-cache-dir transformers==4.49.0

# No source code is copied into the image.
# predict.sh bind-mounts the example directory at /workspace at runtime,
# so predict.py and upstream/ are available via the mount.
