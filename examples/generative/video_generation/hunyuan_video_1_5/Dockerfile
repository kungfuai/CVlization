FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Install flash-attn (optional, for faster attention)
# Note: This requires devel image and longer build time
# RUN pip install flash-attn --no-build-isolation

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default command
CMD ["python3", "predict.py", "--help"]
