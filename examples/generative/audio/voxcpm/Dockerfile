# VoxCPM1.5: Tokenizer-free Text-to-Speech
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
# Disable torch.compile (requires C compiler not in runtime image)
ENV TORCHDYNAMO_DISABLE=1

# Default command
CMD ["python", "predict.py"]
