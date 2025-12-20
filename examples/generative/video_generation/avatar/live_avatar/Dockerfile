FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Install flash-attn from pre-built wheel (much faster than compiling)
# Source: https://flashattn.dev/ - for CUDA 12.8, PyTorch 2.9, Python 3.11
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

WORKDIR /workspace

# Clone LiveAvatar repository
RUN git clone https://github.com/Alibaba-Quark/LiveAvatar.git /workspace/LiveAvatar

# Patch causal_s2v_pipeline.py to avoid accelerate's dispatch mechanism
# which causes tensor device mismatches. Load to CPU first, then move to GPU.
# We remove device_map and add an explicit .to() call.
RUN sed -i 's/device_map=self.device)/)/g' \
    /workspace/LiveAvatar/liveavatar/models/wan/causal_s2v_pipeline.py && \
    sed -i '/self.noise_model.freqs.to(device=self.device)/i\        self.noise_model = self.noise_model.to(self.device)' \
    /workspace/LiveAvatar/liveavatar/models/wan/causal_s2v_pipeline.py

COPY predict.py /workspace/predict.py

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache

WORKDIR /workspace
CMD ["python3", "predict.py", "--help"]
