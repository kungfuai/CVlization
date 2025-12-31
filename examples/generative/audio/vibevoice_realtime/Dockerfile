# VibeVoice-Realtime-0.5B: Real-time TTS inference
# PyTorch 2.9.1 with CUDA 12.8 for Blackwell GPU support
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone and install VibeVoice
RUN git clone https://github.com/microsoft/VibeVoice.git /opt/VibeVoice && \
    cd /opt/VibeVoice && \
    pip install --no-cache-dir -e .

# Install flash-attn for better performance (optional but recommended)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, will use eager attention"

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Default command
CMD ["python", "predict.py"]
